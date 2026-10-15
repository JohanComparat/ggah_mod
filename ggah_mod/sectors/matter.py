r"""The matter field: six components, adding to one by construction.

Everything that lenses, as one tracer:

.. math::

    W_m(k|M) = \frac{M}{\bar\rho_m}\Big[
        f_{\rm coll}\,\tilde u_{\rm DM}
      + f_{\rm hot}\,\tilde u_{\rm gas}
      + f_\star^{\rm cen}
      + f_\star^{\rm sat}\,\tilde u_{\rm sat}
      + f_{\rm cold}\,\tilde u_{\rm sat}
      + f_{\rm ej}\,\tilde u_{\rm ej}\Big]

Each :math:`\tilde u \to 1` as :math:`k \to 0`, which
:func:`~ggah_mod.halos.profiles.profile_uk_gl` gives by construction rather
than by a guard, and the six fractions sum to one identically, so

.. math::  W_m(k \to 0|M) = M/\bar\rho_m

exactly: mass conservation is a property of the algebra.

Six components
--------------

A budget that cannot fail to close constrains the sum, never the split, so a
three-component version closed as well while misplacing three things, which
the cosmic census showed.

**Collisionless matter.** The three-component ``f_cdm`` was
:math:`1 - f_{\rm gas} - f_\star`, the collisionless matter plus the halo's
missing baryons, on :math:`\tilde u_{\rm DM}`: the expelled gas went back on an
NFW profile, which around :math:`10^{12}\,M_\odot/h` is most of the halo's
baryons.
``cdm`` in this package means cold dark matter and nothing else, so
:func:`f_collisionless` names the halo's dark matter as a quantity in its own
right rather than as a remainder.

**Gas or baryons.** Every route in :mod:`~ggah_mod.sectors.energetics`
saturates at :math:`f_b^{\rm cosmic}`, which reads as "every baryon retained",
and a caller that adds :math:`f_\star` on top overcounts.
At the default parameters and z = 0, the energy route then puts
:math:`f_\star + f_{\rm gas}` above :math:`f_b^{\rm cosmic}` on 56 of the 256
mass nodes of ``DIFFERENTIABLE``, worst by a factor :math:`1.041` at
:math:`10^{16}\,M_\odot/h`: a halo holding 4% more baryons than exist.
:class:`BaryonSplit` therefore has two constructors that take the two
quantities by name: :meth:`~BaryonSplit.from_retained` for the energy closure,
which predicts baryons, and :meth:`~BaryonSplit.from_hot` for the parametrised
forms, which were calibrated against gas.

Differences from the predecessor
--------------------------------

* **Two-halo term.** ``P_gm^{2h}`` assigned the entire two-halo term to the
  CDM component "so that cdm + b = total", in both the numpy and the JAX
  paths.
  Here the split lives in the weights, and layer 4's generic integral
  inherits it.
* **Stellar term.** ``log10_M_star_cen`` was added to :math:`\Delta\Sigma`
  outside the halo model.
  Here :math:`f_\star` sits inside the weight, as a point mass that counts in
  the budget.
* **Legs of one cross-spectrum.** :math:`C_\ell^{\kappa y}` had a total-matter
  NFW on the lensing side and a DPM pressure profile on the :math:`y` side, so
  the same halo had two gas contents.
  One :class:`BaryonSplit` serves both legs.

Total matter density
--------------------

:math:`\bar\rho_m` is :attr:`~ggah_mod.cosmology.parameters.Cosmology.rho_matter`:
total matter, neutrinos included, because that is what lenses.
It is never ``rho_cold``, which haloes form from and which is 0.46% smaller at
the minimal neutrino mass; :func:`cosmic_baryon_fraction` is where the
distinction matters.

Neutrinos
---------

The neutrinos are in no halo.
``dn/dM`` is calibrated against the cold field, so the :math:`M` it counts is
cold mass, :math:`\int M\,(dn/dM)\,dM \to \bar\rho_{cb}`; massive neutrinos
free-stream and do not collapse with it.
The six fractions are therefore fractions of the cold halo mass: the baryons'
ceiling is :math:`\Omega_b/\Omega_{cb}` (:func:`cosmic_baryon_fraction`) and
:func:`f_collisionless` is :math:`\Omega_{cdm}/\Omega_{cb}`, cold dark matter
and nothing else.

The neutrinos enter only through the prefactor.
Normalising cold mass by :math:`\bar\rho_m` rather than :math:`\bar\rho_{cb}`
makes the halo term :math:`(1 - f_\nu)` of the total matter:
:math:`P_{mm}^{1h} \propto (1-f_\nu)^2` and, with the counterterm,
:math:`P_{mm}^{2h} \to (1-f_\nu)^2 P_{cb}` from the halo integral alone.
The rest of the total matter spectrum,
:math:`2f_\nu(1-f_\nu)P_{cb\nu} + f_\nu^2P_{\nu\nu}`, is linear, and layer 4
adds it as a leg on the two-halo amplitude
(:mod:`~ggah_mod.spectra.neutrinos`); this module sets
``neutrino_weight = 1``, so the matter field carries the neutrinos whole.

A share of :math:`1 - \Omega_b/\Omega_m` (CDM plus neutrinos) would put the
neutrinos on the NFW profile of every halo, lower every halo's baryon ceiling
by :math:`1 - f_\nu`, and keep the cosmic census from closing on
:math:`\Omega_b` even over a complete mass function.
The weights' :math:`k \to 0` limit is the same either way, so no budget check
detects it.

Stars of the central galaxy as a point mass
-------------------------------------------

:math:`\tilde u_\star = 1` at every :math:`k`.
A galaxy is not resolved on the scales a halo model works at, so its profile
is a delta function whose transform is flat; this is exact here, and the
stellar term needs no profile parameter.

It is exact for the central galaxy only.
Satellite stars follow the satellite distribution, so they belong in
:math:`w_{\rm extended}` with :math:`\tilde u_{\rm sat}`.
At :math:`M \sim 10^{14}` satellites carry most of the stellar mass, and
leaving them out is invisible to any :math:`k \to 0` check, since the budget
closes whichever component the stars are put in.
It shows up only as a wrong :math:`\Delta\Sigma` at
:math:`R \lesssim 0.3` Mpc/h; hence ``f_star_sat``.

Cold gas rides its own profile, the exponential of
:meth:`~ggah_mod.sectors.coldgas.ColdGasSector.u_k`, whenever the neutral-gas
sector that supplied :math:`f_{\rm cold}` is present: a :class:`MatterField`
built with ``coldgas=`` reads it, and layer 4 hands it the sector's own
parameters as an optional peer.
The gas is in galaxies rather than smooth through the halo, and its profile is
not the satellite stars': :math:`c_{\rm HI} \simeq 30` against a halo
concentration of about 5, so the two transforms part above
:math:`k \sim 3\,h\,{\rm Mpc}^{-1}` (:math:`\tilde u = 0.96` against
:math:`0.20` at :math:`k = 20`, :math:`M = 10^{13}`).
Without that sector the fraction carries no shape, so ``u_cold`` falls back to
:math:`\tilde u_{\rm sat}`.
The effect on :math:`P_{mm}` is small, as the phase is: :math:`2\times10^{-7}`
at :math:`k = 0.1`, :math:`3.6\times10^{-5}` at 1 and :math:`3.2\times10^{-3}`
at 10.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .protocol import TracerWeights

__all__ = ["MatterField", "matter_weights", "BaryonSplit", "f_collisionless",
           "cosmic_baryon_fraction"]


def cosmic_baryon_fraction(cosmo):
    r""":math:`\Omega_b/\Omega_{cb}`: the baryon share of the mass a halo holds.

    The ``f_b_cosmic`` that both :class:`BaryonSplit` constructors and every
    route in :mod:`~ggah_mod.sectors.energetics` take.
    :math:`\Omega_{cb}` and not :math:`\Omega_m`, because the halo mass is cold
    mass and the neutrinos are in no halo (module docstring, "Neutrinos").
    :math:`\Omega_b/\Omega_m` is smaller by :math:`1 - f_\nu`, 0.46% at the
    minimal neutrino mass, and a split built from it leaves
    :meth:`~ggah_mod.sectors.census.BaryonCensus.closure_residual` non-zero.
    Without massive neutrinos the two are the same number.
    """
    return cosmo.Omega_b / cosmo.Omega_cb


def f_collisionless(f_b_cosmic):
    r""":math:`1 - f_b^{\rm cosmic}`: the halo's dark matter.

    Given :func:`cosmic_baryon_fraction`, this is
    :math:`\Omega_{cdm}/\Omega_{cb}`: cold dark matter, with no neutrinos in
    it, because the neutrinos are in no halo.

    It assumes that every halo accreted its cosmic share of baryons and that
    only feedback removed them.
    This is the standard baryonification assumption, and it fails below the
    filtering mass, where haloes never accreted their share.
    Nothing here suppresses it: a suppression would be a model, with a
    parameter and a reason.
    """
    return 1.0 - jnp.asarray(f_b_cosmic)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BaryonSplit:
    r"""How one halo's mass divides, in six named parts that sum to one.

    Built through :meth:`from_retained` or :meth:`from_hot`, never by hand:
    the two differ in which baryon fraction the caller has measured.

    Attributes
    ----------
    f_collisionless : array (NM,)
        :math:`1 - f_b^{\rm cosmic} = \Omega_{cdm}/\Omega_{cb}`.  Cold dark
        matter; the neutrinos are in no halo.

    Every fraction is of the halo's cold mass, the :math:`M` that ``dn/dM``
    counts, so ``f_b_cosmic`` is :func:`cosmic_baryon_fraction`.
    f_hot : array (NM,)
        Hot gas, on the DPM profile, as an X-ray or tSZ measurement sees it.
    f_star_cen, f_star_sat : array (NM,)
        Stellar mass, central and satellite.
    f_cold : array (NM,)
        Neutral gas, HI + H2.  Zero until the cold-gas sector supplies it.
    f_ejected : array (NM,)
        :math:`f_b^{\rm cosmic}` minus everything retained: the baryons pushed
        out of the halo.
        Not clipped: a negative value means the halo has been given more
        baryons than exist, which calls for different parameters, and a
        ``maximum`` would also kill the gradient.
    """

    f_collisionless: jnp.ndarray
    f_hot: jnp.ndarray
    f_star_cen: jnp.ndarray
    f_star_sat: jnp.ndarray
    f_cold: jnp.ndarray
    f_ejected: jnp.ndarray

    # ----------------------------------------------------------------- pytree
    def tree_flatten(self):
        return ((self.f_collisionless, self.f_hot, self.f_star_cen,
                 self.f_star_sat, self.f_cold, self.f_ejected), ())

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    def replace(self, **kw) -> "BaryonSplit":
        """A copy with some fractions changed."""
        return replace(self, **kw)

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_retained(cls, f_b_cosmic, f_retained, f_star_cen=0.0,
                      f_star_sat=0.0, f_cold=0.0) -> "BaryonSplit":
        r"""Build the split from the baryon fraction a halo kept.

        :func:`~ggah_mod.sectors.energetics.f_retained_energy` predicts this
        fraction: the energy closure balances the baryons displaced against
        the binding energy, so what comes out is
        :math:`f_b^{\rm cosmic} - \Delta f_b`, stars included.
        The hot gas is the remainder,
        :math:`f_{\rm hot} = f_{\rm ret} - f_\star - f_{\rm cold}`, and it goes
        negative, unclipped, if the stars and cold gas exceed what was retained.
        """
        f_b = jnp.asarray(f_b_cosmic)
        f_ret = jnp.atleast_1d(jnp.asarray(f_retained))
        f_cen = jnp.atleast_1d(jnp.asarray(f_star_cen))
        f_sat = jnp.atleast_1d(jnp.asarray(f_star_sat))
        f_c = jnp.atleast_1d(jnp.asarray(f_cold))
        return cls(f_collisionless=jnp.broadcast_to(f_collisionless(f_b),
                                                    f_ret.shape),
                   f_hot=f_ret - f_cen - f_sat - f_c,
                   f_star_cen=jnp.broadcast_to(f_cen, f_ret.shape),
                   f_star_sat=jnp.broadcast_to(f_sat, f_ret.shape),
                   f_cold=jnp.broadcast_to(f_c, f_ret.shape),
                   f_ejected=f_b - f_ret)

    @classmethod
    def from_hot(cls, f_b_cosmic, f_hot, f_star_cen=0.0, f_star_sat=0.0,
                 f_cold=0.0) -> "BaryonSplit":
        r"""Build the split from the hot-gas fraction.

        The parametrised forms in :mod:`~ggah_mod.sectors.energetics` were
        calibrated against this fraction (FLAMINGO measures a gas fraction, not
        a baryon fraction), and the DPM's own
        :meth:`~ggah_mod.sectors.gas.HotGasDPM.gas_mass` integral returns it.
        The retained baryons are then :math:`f_{\rm hot} + f_\star + f_{\rm cold}`.

        The parametrised forms saturate at :math:`f_b^{\rm cosmic}`, which under
        this reading counts the stars once too many: their ceiling should be
        :math:`f_b^{\rm cosmic} - f_\star - f_{\rm cold}`.
        This method does not repair it, as the repair belongs to a
        recalibration of the fit; :attr:`f_ejected` goes negative where it
        matters, and :meth:`worst_overdraft` measures by how much.
        """
        f_b = jnp.asarray(f_b_cosmic)
        f_h = jnp.atleast_1d(jnp.asarray(f_hot))
        f_cen = jnp.atleast_1d(jnp.asarray(f_star_cen))
        f_sat = jnp.atleast_1d(jnp.asarray(f_star_sat))
        f_c = jnp.atleast_1d(jnp.asarray(f_cold))
        f_ret = f_h + f_cen + f_sat + f_c
        return cls(f_collisionless=jnp.broadcast_to(f_collisionless(f_b),
                                                    f_h.shape),
                   f_hot=f_h,
                   f_star_cen=jnp.broadcast_to(f_cen, f_h.shape),
                   f_star_sat=jnp.broadcast_to(f_sat, f_h.shape),
                   f_cold=jnp.broadcast_to(f_c, f_h.shape),
                   f_ejected=f_b - f_ret)

    # ------------------------------------------------------------- diagnostics
    @property
    def f_baryon(self):
        r""":math:`f_b^{\rm cosmic}`, recovered from the parts that make it."""
        return (self.f_hot + self.f_star_cen + self.f_star_sat + self.f_cold
                + self.f_ejected)

    @property
    def f_retained(self):
        r"""The baryons still in the halo: everything but :attr:`f_ejected`."""
        return self.f_hot + self.f_star_cen + self.f_star_sat + self.f_cold

    def residual(self):
        r""":math:`\sum_i f_i - 1`, which must vanish.

        Returned rather than asserted, like
        :meth:`MatterField.mass_conservation_residual`, so that the parity
        budget records it as a measured number.
        """
        return (self.f_collisionless + self.f_hot + self.f_star_cen
                + self.f_star_sat + self.f_cold + self.f_ejected) - 1.0

    def worst_overdraft(self):
        r""":math:`\max(-f_{\rm ej})`, positive when a halo is overdrawn.

        Zero or negative means every halo's baryons fit inside its cosmic
        share.
        Positive is the amount by which the worst one does not, as a fraction
        of the halo mass.
        """
        return jnp.max(-self.f_ejected)


def matter_weights(field, split: BaryonSplit, u_gas=None, u_dm=None,
                   u_sat=None, u_ej=None, name="matter", *,
                   u_cold=None) -> TracerWeights:
    r"""Assemble :math:`W_m(k|M)` from a :class:`BaryonSplit`.

    Parameters
    ----------
    field : HaloField
    split : BaryonSplit
        The six fractions, traced throughout.
        A pair of arrays could not say whether its second argument was gas or
        baryons, which differ by :math:`f_\star` on most of the mass range.
    u_gas : array (Nk, NM), optional
        The hot gas profile's normalised transform.
        Defaults to the dark-matter one, i.e. gas that traces the dark matter:
        the no-feedback-shape limit, which also keeps this module testable
        without the gas sector.
    u_dm : array (Nk, NM), optional
        Defaults to the field's NFW transform.
    u_sat : array (Nk, NM), optional
        Carries the satellite stars, and the cold gas unless ``u_cold`` is
        given.  Defaults to the dark-matter profile, i.e. satellites tracing
        the mass.
    u_cold : array (Nk, NM), optional
        The neutral gas, on the same transform as the standalone ``coldgas``
        tracer: :meth:`~ggah_mod.sectors.coldgas.ColdGasSector.u_k`.
        :class:`MatterField` passes it when it was built with a ``coldgas``
        sector.
        Defaults to ``u_sat`` when nothing supplies one, because the fraction
        does not carry a shape and this layer does not invent one.
    u_ej : array (Nk, NM), optional
        The ejected gas.
        Defaults to the dark-matter profile, which is knowingly the wrong
        physics: the three-component version had no ejected term and carried
        these baryons on :math:`\tilde u_{\rm DM}`, so with this default the
        six-component weights differ from it for one reason only (the stars
        are no longer counted inside the gas fraction), and the profile change
        is a separate, opt-in step.
        Pass :func:`~ggah_mod.halos.profiles.ejected_uk` for the physics.

    Returns
    -------
    TracerWeights
        Continuous (``discrete=False``): the matter field is not a countable
        population, so its one-halo auto-spectrum has no self-pair to exclude.
    """
    u_dm = field.u_nfw() if u_dm is None else jnp.asarray(u_dm)
    u_gas = u_dm if u_gas is None else jnp.asarray(u_gas)
    u_sat = u_dm if u_sat is None else jnp.asarray(u_sat)
    u_ej = u_dm if u_ej is None else jnp.asarray(u_ej)

    m_over_rho = field.m / field.rho_matter                       # (NM,)

    # The CENTRAL's stars are the point component: u_star = 1 at every k,
    # exactly.  Everything else follows a profile, and `f_collisionless` is a
    # fraction of the *total* matter density, so the budget closes identically
    # whichever component a given baryon is put in -- which is why the split
    # has to be got right for a reason other than this invariant.
    w_point = m_over_rho * split.f_star_cen                       # (NM,)
    # Two branches rather than `u_cold = u_sat`: (a + b) u and a u + b u are
    # not the same floating-point number, and the shipped weights must not move
    # by a ULP for a caller who passed no neutral-gas profile.
    if u_cold is None:
        sat_and_cold = (split.f_star_sat + split.f_cold) * u_sat
    else:
        sat_and_cold = (split.f_star_sat * u_sat
                        + split.f_cold * jnp.asarray(u_cold))
    w_extended = m_over_rho * (split.f_collisionless * u_dm
                               + split.f_hot * u_gas
                               + sat_and_cold
                               + split.f_ejected * u_ej)          # (Nk, NM)

    return TracerWeights(w_point=w_point, w_extended=w_extended,
                         norm=jnp.asarray(1.0), discrete=False,
                         bias_weight=None, name=name,
                         neutrino_weight=jnp.asarray(1.0))


class MatterField:
    """The matter tracer: a :class:`~ggah_mod.sectors.protocol.Sector`.

    Holds no grid and no parameters of its own: it composes the fractions
    other sectors supply, as a peer that reads its siblings' outputs.

    ``coldgas`` is the one sibling it may be built with, and it is held for the
    reason the AGN sector holds its galaxies: the profile the neutral gas is put
    on must be the profile the ``coldgas`` tracer uses, or :math:`P_{mm}` and
    :math:`P_{\rm HI}` would describe two different gas distributions.
    Layer 4 supplies that sector's parameters as an optional peer and refuses a
    spectrum whose ``coldgas`` sector is a different instance.
    """

    name = "matter"
    differentiable = True

    def __init__(self, coldgas=None):
        self.coldgas = coldgas

    @property
    def galaxies(self):
        """The galaxy sector the neutral gas reads, or None.

        Exposed for layer 4's identity check: a per-galaxy neutral gas
        (``catinella18``) puts HI where the galaxy sector puts galaxies, so a
        matter spectrum carrying a different galaxy sector would describe two
        galaxy populations at once.
        """
        return getattr(self.coldgas, "galaxies", None)

    def weights(self, field, params, coldgas_params=None,
                galaxies_params=None) -> TracerWeights:
        r""":math:`W_m(k|M)` for the split in ``params``.

        ``params`` must carry ``split``, a :class:`BaryonSplit`.
        Profiles, ``u_dm`` included, are optional and default as
        :func:`matter_weights` documents.
        """
        if "split" not in params:
            raise ValueError(
                "the matter sector needs `split`, a BaryonSplit.  It used to "
                "take `f_gas` and `f_star`, which could not say whether the "
                "first meant hot gas or all retained baryons -- and the two "
                "differ by f_star, which is up to 2.8% of the halo mass.  Build "
                "one with BaryonSplit.from_retained (the energy closure) or "
                "BaryonSplit.from_hot (the parameterised forms).")
        return matter_weights(field, params["split"],
                              u_gas=params.get("u_gas"),
                              u_dm=params.get("u_dm"),
                              u_sat=params.get("u_sat"),
                              u_ej=params.get("u_ej"),
                              u_cold=self._u_cold(field, params, coldgas_params,
                                                  galaxies_params))

    def _u_cold(self, field, params, coldgas_params, galaxies_params=None):
        """The neutral gas's transform: the caller's, the sector's, or None.

        An explicit ``u_cold`` wins.
        Otherwise the sector this field was built with is asked, at its own
        parameters.
        None leaves :func:`matter_weights` on its documented fallback.
        """
        if params.get("u_cold") is not None:
            return params["u_cold"]
        if self.coldgas is None or coldgas_params is None:
            # No parameters, no profile: the sector's shape is a function of
            # them, and inventing a default here would be the model choice this
            # layer refuses to make.  It is also what lets
            # `spectra.tracers.matter_suppression` build its dark-matter-only
            # reference by dropping the peer along with the profiles.
            return None
        return self.coldgas.u_k(field, coldgas_params,
                                galaxies_params=galaxies_params)

    @staticmethod
    def mass_conservation_residual(field, weights) -> jnp.ndarray:
        r""":math:`W_m(k\to0|M)\,\bar\rho_m/M - 1`, which must vanish.

        Returned rather than asserted, so that the parity budget records the
        layer's strongest invariant as a measured number.
        """
        return weights.at_large_scales() * field.rho_matter / field.m - 1.0
