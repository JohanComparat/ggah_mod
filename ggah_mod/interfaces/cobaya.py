r"""``ggah_mod`` as a cobaya ``Theory``, so any cobaya likelihood can drive it.

``PLAN.md`` item **E1**, and the benchmark's seventh recommendation -- the one
it scored *low* cost because "the parameter and unit mapping is written and
tested in ``ggah_bench/adapters/cobaya.py``".  That adapter drives cobaya's
CAMB and CLASS wrappers *from* the benchmark; this is the other direction, and
it is the direction that makes the package usable without a wrapper of one's
own.

The one line that matters
-------------------------

cobaya's cosmology parameters are :math:`H_0`, :math:`\omega_b = \Omega_bh^2`,
:math:`\omega_c = \Omega_ch^2`, :math:`\Sigma m_\nu`.  This package's second
non-negotiable decision is that **`Omega_m` contains the neutrinos** and
``Omega_cdm`` is derived from it.  So the conversion is

.. math::

    \Omega_m = \frac{\omega_b + \omega_c + \omega_\nu}{h^2},
    \qquad \omega_\nu = \Omega_{\nu,\rm matter}\,h^2

and **not** :math:`(\omega_b+\omega_c)/h^2`, which is the form every other
interface writes and which omits the neutrinos entirely.  At
:math:`\Sigma m_\nu = 0.06` eV that is 0.0014 in :math:`\Omega_m` -- half a per
cent, far too small to look like a bug and far too large to ignore in a
posterior.  ``tests/test_cobaya_interface.py`` pins the round trip.

:math:`\Omega_{\nu,\rm matter}` is
:attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_nu_matter`, the
matter-like part of the relic neutrino density, and **not** the familiar
:math:`\Sigma m_\nu/93.14\,{\rm eV}`: it has to be the density
``Cosmology.Omega_cdm`` subtracts on the way back out, and closing with the
93.14 convention instead put ``omch2`` back 3e-6 away from where it came in.
The two differ by 0.46 per cent at 0.06 eV (see
:func:`cosmology_from_cobaya`).

The amplitude is ``logA``
-------------------------

cobaya's two Boltzmann wrappers spell it differently -- the CAMB one takes
``logA`` and derives ``As``, the CLASS one refuses ``logA`` and wants ``A_s``,
which is a finding of the benchmark rather than a choice of anyone's.  This
theory takes **``logA``**, because it is what Planck likelihoods sample and
because :math:`\ln(10^{10}A_s)` is this package's one amplitude.  ``As``,
``sigma8`` and ``S8`` are **refused as inputs**, which is decision 1: they are
computed from the spectrum, and a cosmology carrying both is over-determined.

What it provides, and what it refuses
-------------------------------------

The background and the **linear** spectrum, plus the halo field a halo-model
likelihood actually wants -- which is the reason to drive this package rather
than a Boltzmann code.  ``nonlinear=True`` **raises**: there is no halofit here,
and returning the linear spectrum for a non-linear request is the silent kind of
wrong.

**Curvature is accepted**, and it was not always.  This module shipped one commit
before ``PLAN.md`` item **E2**, and it refused :math:`\Omega_k \ne 0` for the
reason a refusal is better than a default: ``Omega_de`` was derived from
flatness, so a package that took the parameter would have ignored it, and a
posterior on a parameter nothing reads is worse than no posterior.  E2 made
``Omega_de`` close against :math:`\Omega_k` and carried it into both Boltzmann
backends, so the refusal expired.  ``omk`` is now a declared parameter with a
default of ``0.0`` -- a value rather than ``None``, so cobaya treats it as
optional and a flat run need not mention it.
"""

from __future__ import annotations

import numpy as np
from cobaya.theory import Theory

from ..backend import DIFFERENTIABLE, resolve_backend
from ..cosmology import Cosmology
from ..cosmology.amplitude import sigma8
from ..cosmology.background import (
    angular_diameter_distance, comoving_distance, hubble_e,
)
from ..cosmology.drag import r_drag
from ..cosmology.growth import f_sigma8, growth_factor
from ..cosmology.power import make_pk

__all__ = ["GgahMod", "cosmology_from_cobaya", "REFUSED"]

#: Inputs this theory refuses, and why.  Each is a decision, not a limitation.
REFUSED = {
    "As": "the amplitude is `logA`; `As` and `logA` together are two spellings "
          "of one number and which wins would depend on the caller",
    "sigma8": "sigma8 is *derived* from the spectrum here (decision 1). A "
              "cosmology carrying both an amplitude and a sigma8 is "
              "over-determined; use `ln10A_s_for_sigma8` to solve for one",
    "S8": "as `sigma8`",
}


def cosmology_from_cobaya(**p) -> Cosmology:
    r"""cobaya's cosmological parameters, as a :class:`Cosmology`.

    Separate from the :class:`Theory` so it can be tested, and used, without
    cobaya running -- the conversion is the part worth checking and it does not
    need a sampler to check it.
    """
    bad = sorted(k for k in REFUSED if p.get(k) is not None)
    if bad:
        raise ValueError("; ".join(f"{k!r}: {REFUSED[k]}" for k in bad))

    h = float(p["H0"]) / 100.0
    omega_k = float(p.get("omk", p.get("omegak", 0.0)))
    omega_b = float(p["ombh2"])
    omega_c = float(p["omch2"])
    sum_mnu = float(p.get("mnu", 0.0))
    # A theory *setting* rather than a sampled parameter, and deliberately not
    # in the `params` block: large-scale structure constrains `sum_mnu` and not
    # how it is divided, so the ordering carries no likelihood gradient and
    # there is nothing for a chain to explore.  Listing it as a parameter would
    # let cobaya vary it, which is the one thing it must not do.  Comparing two
    # orderings means running twice.
    nu_hierarchy = p.get("nu_hierarchy", "normal")

    # The load-bearing line.  `omch2` is *cold dark matter only*; this package's
    # Omega_m contains the neutrinos, so their density is added here rather
    # than left out -- see the module docstring for what leaving it out costs.
    #
    # It has to be the *same* neutrino density that `Cosmology.Omega_cdm`
    # subtracts, or the round trip is not one: that property uses
    # `Omega_nu_matter`, the matter-like part of the relic neutrino density, and
    # closing here with the 93.14 convention instead put `omch2` back 3e-6
    # away from where it came in.  `Omega_nu_matter` depends only on `h`,
    # `T_cmb` and `sum_mnu`, so a probe cosmology reads it without needing the
    # `Omega_m` this line is solving for.
    #
    # The hierarchy is carried into the probe as well.  It happens not to
    # matter -- `Omega_nu_matter` is linear in the masses, so it depends only
    # on the sum and is bit for bit invariant under the split -- but relying on
    # that invariance from another module is exactly the coupling that breaks
    # the day someone makes it nonlinear.  It is one keyword.
    probe = Cosmology.create(h=h, sum_mnu=sum_mnu, nu_hierarchy=nu_hierarchy)
    omega_nu = float(probe.Omega_nu_matter) * h ** 2
    omega_m = (omega_b + omega_c + omega_nu) / h ** 2

    return Cosmology.create(
        h=h, Omega_m=omega_m, Omega_b=omega_b / h ** 2,
        n_s=float(p["ns"]), ln10A_s=float(p["logA"]),
        sum_mnu=sum_mnu,
        Omega_k=omega_k, nu_hierarchy=nu_hierarchy,
        w0=float(p.get("w", -1.0)), wa=float(p.get("wa", 0.0)))


class GgahMod(Theory):
    r"""A cobaya ``Theory`` backed by ``ggah_mod``.

    ``yaml`` usage::

        theory:
          ggah_mod.interfaces.cobaya.GgahMod:
            backend: differentiable
            pk: emu_pk

    Set ``backend`` to ``accurate`` for a Boltzmann solve; the flavour decides
    the grids and the linear-P(k) backend exactly as it does everywhere else in
    the package, so a likelihood cannot silently get a coarser model than it
    thinks.
    """

    #: The flavour name, resolved through :func:`~ggah_mod.backend.resolve_backend`.
    backend: str = "differentiable"
    #: Which linear-P(k) backend; ``None`` takes the flavour's declared one,
    #: which is the only value that cannot disagree with it.
    pk: str | None = None
    #: Redshifts the halo field is built at, for a halo-model likelihood.
    z_halo: tuple = ()
    #: Which neutrino mass ordering, as a **setting** and not a sampled
    #: parameter.
    #:
    #: Large-scale structure constrains :math:`\Sigma m_\nu` and not how it is
    #: divided, so the ordering carries no likelihood gradient: it is chosen
    #: once per run and comparing two means running twice.  Putting it in
    #: ``params`` would let cobaya vary it, which is the one thing it must not
    #: do.
    #:
    #: The default is ``"normal"``, which is the physics.  It used to be
    #: unavailable on the differentiable flavour -- ``emu_pk`` 1.x was trained on
    #: degenerate species and refused a split -- but 2.0.0 carries
    #: ``nu_r1``/``nu_r2`` and answers for any ordering, so **both flavours take
    #: it since 2026-09-10** and a differentiable run no longer has to say
    #: ``nu_hierarchy: degenerate`` to work.
    #:
    #: **What the default does still cost, and it is a live trap.**  A normal
    #: ordering has *no solution* below 0.058993 eV: the two measured splittings
    #: fix a floor, and :class:`~ggah_mod.cosmology.parameters.Cosmology`
    #: refuses beneath it.  Exactly zero is allowed, because every ordering
    #: returns three zero masses there.  So a chain sampling ``mnu`` on the
    #: usual :math:`[0, 0.5]` prior raises part-way through -- in the *interior*
    #: of the prior, from just above zero to 0.059, with both ends fine -- and
    #: it raises rather than rejecting the point.  ``emu_hmf`` 1.2.0 met the
    #: same floor from the other side and pins ``"degenerate"`` for it, after an
    #: unpinned construction refused 393 of its 2000 design points.  Say
    #: ``nu_hierarchy: degenerate`` for any run whose neutrino prior reaches
    #: below 0.059 eV, or start the prior above the floor.
    nu_hierarchy: str = "normal"

    #: ``omk`` carries a value rather than ``None`` so it is *optional*: a
    #: ``None`` here makes cobaya demand the parameter be sampled or fixed by
    #: the caller, which would force every flat analysis to say so.  Zero is the
    #: default the package already had, and a curvature run overrides it.
    params = {"H0": None, "ombh2": None, "omch2": None,
              "logA": None, "ns": None, "mnu": None, "omk": 0.0}

    def initialize(self):
        self._backend = resolve_backend(self.backend)
        self._pk = make_pk(self.pk or self._backend.pk)
        self._k = np.asarray(
            np.logspace(np.log10(self._backend.k_min),
                        np.log10(self._backend.k_max), self._backend.n_k))
        #: Redshifts a likelihood asked ``Pk_grid`` for; see :meth:`must_provide`.
        self._pk_z: tuple = ()
        #: Whether a likelihood asked for ``rdrag``; see :meth:`must_provide`.
        self._want_rdrag = False

    def must_provide(self, **requirements):
        r"""Record what was asked for, which is where the redshifts come from.

        cobaya resolves its dependency graph once, at construction, and a
        quantity's *arguments* arrive here rather than at the accessor.  So
        ``Pk_grid``'s redshifts are the likelihood's, not this block's
        ``z_halo`` -- getting that backwards makes a likelihood asking for one
        redshift silently receive whichever the theory block happened to name,
        with the right shape and the wrong epoch.
        """
        for name, opts in requirements.items():
            if name == "Pk_grid" and opts:
                z = opts.get("z")
                if z is not None:
                    self._pk_z = tuple(float(v) for v in np.atleast_1d(z))
                k_max = opts.get("k_max")
                if k_max is not None and float(k_max) > self._backend.k_max:
                    raise ValueError(
                        f"a likelihood asked for P(k) to k = {float(k_max)} "
                        f"h/Mpc and the {self._backend.name!r} flavour's grid "
                        f"stops at {self._backend.k_max}. Widening it here "
                        f"would change sigma(M) too -- `make_field`'s k grid is "
                        f"also its quadrature -- so the flavour is the place to "
                        f"change it.")
            elif name == "rdrag":
                # A BAO likelihood's request arrives here, not in
                # `output_params`, which holds only the YAML's derived block.
                self._want_rdrag = True
            elif name == "ggah_fields" and opts:
                z = opts.get("z")
                if z is not None:
                    self.z_halo = tuple(float(v) for v in np.atleast_1d(z))

    def get_can_provide(self):
        return ["ggah_cosmology", "Hubble", "comoving_radial_distance",
                "angular_diameter_distance", "Pk_grid", "sigma8_z",
                "fsigma8", "growth_factor", "ggah_fields"]

    def get_can_provide_params(self):
        r"""``rdrag``, the BAO sound horizon in **Mpc** -- the name cobaya's BAO
        likelihoods (``bao.desi_dr2`` and the rest) ask a theory for."""
        return ["rdrag"]

    def calculate(self, state, want_derived=True, **params):
        cosmo = cosmology_from_cobaya(nu_hierarchy=self.nu_hierarchy, **params)
        state["ggah_cosmology"] = cosmo
        if self.z_halo:
            # Built here rather than in the accessor, so the halo chain is
            # computed once per parameter point.  cobaya caches `state`; an
            # accessor that built its own would repeat a Boltzmann solve for
            # every likelihood that asked, which is the defect `make_model`
            # already documents about `fields_at`.
            from ..halos.field import make_fields

            state["ggah_fields"] = make_fields(
                cosmo, self._backend, pk=self._pk,
                z=tuple(float(v) for v in self.z_halo))
        if want_derived:
            state["derived"] = {
                "sigma8": float(sigma8(
                    np.asarray(self._pk.pk(self._k, 0.0, cosmo)), self._k)),
                "Omega_m": float(cosmo.Omega_m),
            }
            # Only when a likelihood asked: `r_drag` refuses a cosmology outside
            # the box its drag-redshift fit was calibrated on, and a chain that
            # never uses r_d must not be stopped by that.  In Mpc, cobaya's unit.
            if self._want_rdrag or "rdrag" in self.output_params:
                state["derived"]["rdrag"] = float(r_drag(cosmo)) / float(cosmo.h)
        return True

    # -- the accessors cobaya looks for ------------------------------------
    def _cosmo(self):
        return self.current_state["ggah_cosmology"]

    def get_ggah_cosmology(self):
        """The :class:`Cosmology` itself, for a likelihood that wants the
        package rather than a table of numbers out of it."""
        return self._cosmo()

    def get_Hubble(self, z, units="km/s/Mpc"):
        c = self._cosmo()
        h_z = np.asarray(hubble_e(np.asarray(z), c)) * 100.0 * float(c.h)
        if units == "km/s/Mpc":
            return h_z
        if units == "1/Mpc":
            from ..cosmology.constants import C_KM_S
            return h_z / C_KM_S
        raise ValueError(f"unknown units {units!r}; expected 'km/s/Mpc' or "
                         f"'1/Mpc'")

    def get_comoving_radial_distance(self, z):
        r"""In **Mpc**, not Mpc/h.

        cobaya's convention, and the opposite of this package's -- which is the
        conversion the benchmark calls "a factor of h between two codes is a 30
        per cent error that reads as a physics result".
        """
        c = self._cosmo()
        return np.asarray(comoving_distance(np.asarray(z), c)) / float(c.h)

    def get_angular_diameter_distance(self, z):
        """In Mpc; see :meth:`get_comoving_radial_distance`."""
        c = self._cosmo()
        return np.asarray(
            angular_diameter_distance(np.asarray(z), c)) / float(c.h)

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"),
                    nonlinear=False):
        r"""``(k [1/Mpc], z, P [Mpc^3])`` -- cobaya's units, not this package's.

        ``var_pair`` selects the total-matter or the cold spectrum, which are
        separately named here for the reason ``rho_cold`` and ``rho_matter``
        are: halos form from one and lensing sees the other.
        """
        if nonlinear:
            raise ValueError(
                "ggah_mod has no non-linear P(k) backend -- there is no "
                "halofit or HMcode here, and the halo model is layer 4 rather "
                "than a correction to layer 1. Returning the linear spectrum "
                "for a nonlinear=True request would be the silent kind of "
                "wrong, so this refuses instead. PLAN.md item E5.")
        cold = var_pair in (("delta_nonu", "delta_nonu"), ("delta_cb", "delta_cb"))
        c = self._cosmo()
        z = np.atleast_1d(np.asarray(self._pk_z or (0.0,), dtype=float))
        p = np.asarray(self._pk.pk(self._k, z, c, variant="cb" if cold else "total")
                       if _takes_variant(self._pk) else self._pk.pk(self._k, z, c))
        h = float(c.h)
        return self._k * h, z, np.atleast_2d(p) / h ** 3

    def get_sigma8_z(self, z):
        c = self._cosmo()
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.asarray([
            float(sigma8(np.asarray(self._pk.pk(self._k, float(zz), c)),
                         self._k)) for zz in z])

    def get_fsigma8(self, z):
        c = self._cosmo()
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.asarray([float(f_sigma8(float(zz), c, self._pk)) for zz in z])

    def get_growth_factor(self, z):
        c = self._cosmo()
        return np.asarray(growth_factor(np.asarray(z), c, self._pk))

    def get_ggah_fields(self, z=None):
        """The :class:`~ggah_mod.halos.field.HaloField` set, which is the point.

        A likelihood that only wanted a background and a linear spectrum would
        be better served by CAMB.  This is what makes driving ``ggah_mod``
        worth the adapter: one call gives the halo chain at every requested
        redshift, built from the same cosmology the background came from.
        """
        if z is None:
            cached = self.current_state.get("ggah_fields")
            if cached is not None:
                return cached
            raise ValueError(
                "no redshifts: ask for `ggah_fields: {z: [...]}` in the "
                "likelihood's requirements, or set `z_halo` on the theory "
                "block, or pass `z=` here. A halo field is one epoch per "
                "object and there is no default -- which one is a property of "
                "the sample.")

        from ..halos.field import make_fields

        return make_fields(self._cosmo(), self._backend, pk=self._pk,
                           z=tuple(float(v) for v in np.atleast_1d(z)))


def _takes_variant(pk) -> bool:
    import inspect

    try:
        return "variant" in inspect.signature(pk.pk).parameters
    except (TypeError, ValueError):     # pragma: no cover - exotic callables
        return False
