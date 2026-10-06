r"""The registry: a :class:`~ggah_mod.spectra.spec.Component` to a
:class:`~ggah_mod.sectors.protocol.TracerWeights`, and back out as a spectrum.

Layer 4 never learns which sector it is talking to -- :mod:`~ggah_mod.spectra.pk`
names none.  *Somewhere* has to, though, because the sectors' ``weights()``
signatures differ, and this is that place: a table of small adapters, and nothing
else.

**Sector instances are supplied, not constructed here.**  ``GalaxySector`` needs
a model name, ``HotGasDPM`` needs a cooling function, ``AgnSector`` needs its
grids -- all *model choices*.  Building them behind the caller would put a
default HOD into a gas-only spectrum, which is the exact coupling the layer-3
peer structure exists to prevent, and it is why
:func:`~ggah_mod.halos.field.make_field` refuses to default its ``pk`` too.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..sectors.protocol import TracerWeights
from .pk import PowerSpectrum, pk_cross
from .spec import Component, Overlap, PkOptions, TracerSpec, overlap_of

__all__ = ["TRACERS", "PEERS", "build_weights", "resolve", "spectrum",
           "SECTOR_VIEWS", "matter_suppression"]


def _galaxies(sector, component, field, params):
    return sector.weights(field, params, view=component.view or "total")


def _gas(sector, component, field, params, *, agn_params=None,
         galaxies_params=None, coldgas_params=None):
    """The hot gas, optionally carrying a feedback budget.

    The peers are optional and that is the contract: a gas spectrum with no
    galaxy sector in it is the case this package exists to make possible, so
    the amplitudes stay the sector's own free parameters unless a caller
    supplies the two sectors a budget needs.
    """
    return sector.weights(field, params, view=component.view or "pressure",
                          agn_params=agn_params,
                          galaxies_params=galaxies_params,
                          coldgas_params=coldgas_params)


def _agn(sector, component, field, params, galaxy_params):
    view = component.view or "counts"
    if view == "counts":
        return sector.weights(field, params, galaxy_params)
    if view == "emission":
        return _agn_emission(sector, component, field, params, galaxy_params)
    if view in _AGN_BRANCH_VIEWS:
        return _agn_emission(sector, component, field, params, galaxy_params,
                             branch=_AGN_BRANCH_VIEWS[view])
    raise ValueError(f"unknown AGN view {view!r}; expected one of "
                     f"{SECTOR_VIEWS['agn']}")


#: The AGN emission split by obscuration, one view per branch (0.9.3).  The two
#: add to ``"emission"``; they exist because a detector's energy-conversion
#: factor differs between the branches, by up to 1e6 in a 100 eV band at the
#: soft end, so counts are a sum of two ECF-weighted luminosities and never one.
_AGN_BRANCH_VIEWS = {"emission_unobscured": "unobscured",
                     "emission_obscured": "obscured"}


def _agn_emission(sector, component, field, params, galaxy_params,
                  branch=None):
    r"""AGN X-ray emission: a **discrete** population, luminosity-weighted.

    ``discrete=True`` is the load-bearing part.  Treating the AGN emission as a
    continuous field is what the predecessor's ``forward_jax`` does -- it forms
    ``Xtot = X_gas + X_agn`` and squares it -- while its own
    ``_pk_tables_XX`` excludes the central self-pair with the comment "no
    central self-pair".  Two implementations of one spectrum, disagreeing at the
    :math:`1/N_c` level.  Here the flag says which, once.

    The weights are :meth:`~ggah_mod.sectors.agn.AgnSector.emission_weights`.
    With no band they are the hard band, 2--10 keV.  With one, each AGN's
    energy arriving in it, through the sector's photon index, its column and
    its obscured fraction -- which is why a band needs a sector built with
    ``obscuration="split"``, and the sector says so if it was not.

    Until 0.9.2 this adapter refused a band: the pieces existed but entered
    only the selection, and the emitted luminosity was never carried into
    another band.  A photon map is the case that needed it -- a
    cross-correlation with 0.5--2 keV events -- and such a map also wants the
    sector's ``selection="none"``, because it counts every AGN's photons.  That
    is the sector's model choice and is not made here.
    """
    if component.band is not None:
        b = component.band
        return sector.emission_weights(field, params, galaxy_params,
                                       band=(b.emin, b.emax), frame=b.frame,
                                       branch=branch)
    return sector.emission_weights(field, params, galaxy_params,
                                   branch=branch)


def _matter(sector, component, field, params, coldgas_params=None,
            galaxies_params=None):
    return sector.weights(field, params, coldgas_params=coldgas_params,
                          galaxies_params=galaxies_params)


def _alignments(sector, component, field, params, coldgas_params=None,
                galaxies_params=None):
    """Like ``_matter``: the alignment sector takes the whole params block,
    because it needs the matter split, an ``IaParams`` and ``D(z)`` together."""
    return sector.weights(field, params, coldgas_params=coldgas_params,
                          galaxies_params=galaxies_params)


def _ejecta(sector, component, field, params):
    return sector.weights(field, params, view=component.view or "mass")


def _coldgas(sector, component, field, params, galaxies_params=None):
    return sector.weights(field, params, view=component.view or "mass",
                          galaxies_params=galaxies_params)


#: ``sector name -> adapter(sector_instance, component, field, params)``.
TRACERS = {
    "galaxies": _galaxies,
    "gas": _gas,
    "agn": _agn,
    "matter": _matter,
    "alignments": _alignments,
    "ejecta": _ejecta,
    "coldgas": _coldgas,
}

#: The views each sector answers to, for error messages and for a resolver.
SECTOR_VIEWS = {
    "galaxies": ("total", "cen", "sat"),
    "gas": ("pressure", "xray", "mass", "density"),
    "agn": ("counts", "emission", "emission_unobscured", "emission_obscured"),
    "matter": ("",),
    "alignments": ("",),
    "ejecta": ("mass", "density"),
    "coldgas": ("hi", "h2", "mass"),
}


def build_weights(component: Component, field, sectors: dict, params: dict):
    """One component's :class:`TracerWeights`.

    Parameters
    ----------
    component : Component
    field : HaloField
    sectors : dict
        ``sector name -> instance``.  Supplied, never constructed here.
    params : dict
        ``sector name -> that sector's parameters``.
    """
    if component.sector not in TRACERS:
        raise ValueError(f"unknown sector {component.sector!r}; expected one "
                         f"of {sorted(TRACERS)}")
    if component.sector not in sectors:
        raise ValueError(
            f"component {component.label()!r} needs the {component.sector!r} "
            f"sector, which was not supplied.  Layer 4 does not build one: "
            f"every sector carries a model choice, and defaulting it would put "
            f"an unasked-for model into the spectrum.")
    if component.sector not in params:
        raise ValueError(
            f"component {component.label()!r} needs {component.sector!r} "
            f"parameters, which were not supplied")
    sector = sectors[component.sector]
    args = (sector, component, field, params[component.sector])
    for peer in PEERS.get(component.sector, ()):
        args = args + (_peer_params(component, sector, sectors, params, peer),)
    kw = {}
    for peer in OPTIONAL_PEERS.get(component.sector, ()):
        if peer in params:
            kw[f"{peer}_params"] = _peer_params(
                component, sector, sectors, params, peer, required=False)
    return TRACERS[component.sector](*args, **kw)


def _peer_params(component, sector, sectors, params, peer, *,
                 required: bool = True):
    r"""The galaxy parameters an AGN component is evaluated with.

    The AGN chain starts from the galaxy sector's stellar masses
    (:class:`~ggah_mod.sectors.agn.AgnSector` is built with one), so an AGN
    component needs that sector's parameters even in a spectrum with no galaxy
    component -- and when the spectrum *does* carry a galaxy sector, it must be
    the very instance the AGN sector reads.  Two would put two stellar-mass
    relations into one spectrum: the AGN sector used to evaluate ``zu15`` at a
    private copy of the fiducial parameters, equal to the galaxy sector's only
    while nobody varied them.
    """
    if peer not in params:
        if not required:
            return None
        raise ValueError(
            f"component {component.label()!r} needs {peer!r} parameters as "
            f"well as its own: the AGN chain starts from the galaxy sector's "
            f"stellar masses, so it is evaluated at that sector's parameters "
            f"even when the spectrum has no galaxy component.")
    owned = getattr(sector, peer, None) or getattr(sector, f"{peer[:-1]}", None)
    if peer in sectors and owned is not None and sectors[peer] is not owned:
        cls = type(sector).__name__
        what = {"galaxies": f"the galaxies and the "
                            f"{getattr(sector, 'name', cls)} would be given "
                            f"two different stellar masses",
                "coldgas": "the neutral gas would sit on two different "
                           "profiles in one spectrum"}.get(
            peer, f"one spectrum would carry two {peer!r} models")
        raise ValueError(
            f"the spectrum's {peer!r} sector is not the one this {cls} was "
            f"built with, so {what}.  Build it from the same instance: "
            f"{cls}(sectors[{peer!r}]).")
    return params[peer]


#: ``sector name -> peer sector names`` whose parameters its adapter also takes,
#: and **without which it cannot answer at all**.  The one place a sector reads
#: a peer's parameters, kept beside :data:`TRACERS` because this module is
#: already where layer 4 names sectors.
#:
#: A tuple rather than a single name because the AGN sector was the only case
#: when this was written and is not the only one now: the hot gas reads both the
#: AGN sector's black-hole masses and the galaxy sector's stellar masses when it
#: is asked to carry a feedback budget.
PEERS = {
    "agn": ("galaxies",),
}

#: ``sector name -> peer sector names`` its adapter takes **when they are there**.
#:
#: Distinct from :data:`PEERS`, and the distinction is load-bearing.  A gas
#: spectrum with no galaxy sector in it is the thing this package was built to
#: make possible -- ``tests/test_sector_coherence.py`` opens with it -- so the
#: hot gas may not *require* peers the way the AGN chain does.  It takes them
#: when the caller supplies them and falls back to its own free amplitudes when
#: it does not, which is also what lets the coupling land switched off.
OPTIONAL_PEERS = {
    # The neutral gas since 0.9.5: the closure may not expel it.
    "gas": ("agn", "galaxies", "coldgas"),
    # The matter field puts the neutral gas on that sector's own exponential
    # rather than on the satellites' profile, and the alignment field is the
    # matter field scaled -- so both read the sector's parameters when the
    # spectrum carries them.  Optional for the same reason the gas sector's
    # peers are: a matter spectrum with no neutral gas in it is a spectrum this
    # package is meant to be able to make, and it falls back to `u_sat`.
    "matter": ("coldgas", "galaxies"),
    "alignments": ("coldgas", "galaxies"),
    # The per-galaxy neutral gas (`catinella18`) sets each galaxy's HI by its
    # stellar mass, so it reads the galaxy sector's parameters -- and the matter
    # and alignment fields read them too, through the neutral gas's profile.
    # Optional, because the halo-total relations read none.
    "coldgas": ("galaxies",),
}


#: Convenience names for the tracers a caller usually wants.
#:
#: ``"xray"`` is the composite the whole component machinery exists for: hot gas
#: *plus* AGN point sources.  ``"lensing"``, ``"convergence"``, ``"shear"`` and
#: ``"cmb_lensing"`` are **not** here, deliberately -- they are not 3D fields.
#: They are the matter field under different radial kernels, and that belongs to
#: layer 5.  Giving them names here would make ``P_kappa_kappa`` and ``P_mm``
#: two arrays that can drift apart.
_NAMED = {
    "matter": lambda: TracerSpec.one("matter", name="matter"),
    # The intrinsic-shear field, which *is* a 3D field -- unlike the lensing
    # entries this registry refuses below.  Shear is the matter field under a
    # radial kernel and belongs to layer 5; an alignment is a property of the
    # galaxies themselves at their own redshift, so it has a P(k) and a place
    # here.  PLAN.md item E3.
    "alignments": lambda: TracerSpec.one("alignments", name="alignments"),
    "galaxies": lambda: TracerSpec.one("galaxies", "total",
                                       population="galaxies", name="galaxies"),
    # Siblings under `galaxies`: a central is never a satellite, so the two
    # samples are disjoint and their cross-spectrum excludes nothing.  Crossing
    # either with `galaxies` itself is the *nested* case, and is refused.
    "centrals": lambda: TracerSpec.one("galaxies", "cen",
                                       population="galaxies/cen",
                                       name="centrals"),
    "satellites": lambda: TracerSpec.one("galaxies", "sat",
                                         population="galaxies/sat",
                                         name="satellites"),
    "gas": lambda: TracerSpec.one("gas", "mass", name="gas"),
    "ejecta": lambda: TracerSpec.one("ejecta", "mass", name="ejecta"),
    "hi": lambda: TracerSpec.one("coldgas", "hi", name="hi"),
    "coldgas": lambda: TracerSpec.one("coldgas", "mass", name="coldgas"),
    # Every free electron, which is what kSZ and a dispersion measure see: the
    # gas still on the DPM profile *and* the gas that left.  `gas:density` alone
    # is the composite's commonest error -- it is the electrons inside
    # R_Delta, and the ejected component is most of the budget outside it.
    # Both amplitudes are masses, so the sum is an electron field up to the one
    # constant `mu_e m_p`, which cancels in every normalised statistic and is
    # `EjectaSector.electron_count` where it does not.
    "electrons": lambda: TracerSpec("electrons", (
        Component("gas", "density"),
        Component("ejecta", "mass"),
    )),
    "pressure": lambda: TracerSpec.one("gas", "pressure", name="pressure"),
    "sz": lambda: TracerSpec.one("gas", "pressure", name="sz"),
    "agn": lambda: TracerSpec.one("agn", "counts", population="agn",
                                  name="agn"),
    "xray": lambda: TracerSpec("xray", (
        Component("gas", "xray"),
        Component("agn", "emission", population="agn"),
    )),
}


def resolve(name) -> TracerSpec:
    """A :class:`TracerSpec` from a convenience name, or pass one through."""
    if isinstance(name, TracerSpec):
        return name
    if isinstance(name, Component):
        return TracerSpec(name.label(), (name,))
    key = str(name).lower()
    if key not in _NAMED:
        raise ValueError(
            f"unknown tracer {name!r}; expected one of {sorted(_NAMED)}, or a "
            f"TracerSpec.  Lensing, convergence, shear and CMB lensing are "
            f"deliberately absent: they are not 3D fields but the matter field "
            f"under different radial kernels, which is layer 5's business.")
    return _NAMED[key]()


def spectrum(field, a, b, sectors: dict, params: dict, *,
             options: PkOptions = PkOptions(), bnl_table=None,
             transition_params=None,
             overlaps: dict | None = None) -> PowerSpectrum:
    r"""The spectrum of two tracers, expanded over their components.

    .. math::

        P_{ab} = \sum_{i \in a}\sum_{j \in b} P_{ij}

    which for an auto-spectrum is :math:`\sum_i P_{ii} + 2\sum_{i<j}P_{ij}`,
    because every ordered pair is summed and :math:`P_{ij} = P_{ji}`.  That sum
    is the whole reason a tracer is a list: it is what the predecessor's
    ``_pk_tables_XX`` writes out by hand.  Summed over ordered pairs, the X-ray
    auto-spectrum (gas plus AGN) is four :func:`~ggah_mod.spectra.pk.pk_cross`
    calls under two pair rules -- see :mod:`~ggah_mod.spectra.spec`.

    The neutrinos' linear leg rides on the components that carry it
    (:attr:`~ggah_mod.sectors.protocol.TracerWeights.neutrino_weight`), so a
    composite gets it from its matter-like parts and not otherwise: the
    ``electrons`` composite is gas plus ejecta, both built on the cold field,
    and carries none.

    Parameters
    ----------
    transition_params : TransitionParams, optional
        Layer 4's own declared parameter, forwarded to
        :func:`~ggah_mod.spectra.pk.pk_cross` the way ``bnl_table`` is.
        ``None`` takes the declared default.

        It was **not** forwarded for two tiers, which made
        ``k_star_coeff`` unreachable from this entry point -- the one the
        calibration repository and the benchmark both use -- while
        ``pk_cross`` beneath it took it and the paper described the two
        arguments as siblings.  A declared parameter with no path from the
        public API reads as a parameter the package does not have, and
        ``ggah_cal`` concluded exactly that in writing.
    overlaps : dict, optional
        ``(population_a, population_b) -> overlap``, for pairs the labels cannot
        settle by themselves.  Rarely needed: two labelled components already
        answer the question.
    """
    a, b = resolve(a), resolve(b)
    overlaps = overlaps or {}
    for spec in {a.name: a, b.name: b}.values():
        _check_gas_closure(spec, field, sectors, params)
    if options.bnl and bnl_table is None:
        # Once per call, not once per component pair.
        from .bnl import table_for
        bnl_table = table_for(field, options)

    total = None
    for ca in a.components:
        wa = build_weights(ca, field, sectors, params)
        for cb in b.components:
            wb = wa if cb == ca else build_weights(cb, field, sectors, params)
            key = (ca.population, cb.population)
            ov = (overlaps.get(key) or overlaps.get(key[::-1])
                  or _branch_overlap(ca, cb, wa, wb)
                  or overlap_of(ca, cb, a_discrete=wa.discrete,
                                b_discrete=wb.discrete))
            p = pk_cross(field, wa, wb, overlap=ov, options=options,
                         bnl_table=bnl_table,
                         transition_params=transition_params,
                         a=ca.label(), b=cb.label())
            total = p if total is None else PowerSpectrum(
                k=p.k, one_halo=total.one_halo + p.one_halo,
                two_halo=total.two_halo + p.two_halo,
                shot=total.shot + p.shot, a=a.name, b=b.name)
    return PowerSpectrum(k=total.k, one_halo=total.one_halo,
                         two_halo=total.two_halo, shot=total.shot,
                         a=a.name, b=b.name)


#: How far a composite's hot and ejected gas may miss the split's non-stellar
#: baryons, as a fraction of the halo mass, before :func:`spectrum` says so.
#: Well above u(k_min) - 1 (5e-7) and the quadrature, well below a real mismatch
#: (a sigmoid split against the DPM misses by 1e-2).
GAS_CLOSURE_TOL = 1e-4


def _branch_overlap(ca, cb, wa, wb):
    r"""The rule two obscuration branches of one AGN population pair by.

    ``emission_unobscured`` and ``emission_obscured`` partition one population:
    an AGN is obscured or it is not.  Labelled as siblings -- which they must
    be, since no object is in both -- :func:`~ggah_mod.spectra.spec.overlap_of`
    would call them independent samples, and the one-halo term would pair a
    halo's unobscured central with its obscured central, a pair of objects that
    never coexist.  That is :math:`2p_{\rm u}p_{\rm o}` too much in every
    AGN auto-spectrum built from the two branches.  The right rule is the one
    :class:`~ggah_mod.spectra.spec.Overlap` already states for two samples drawn
    from the same one-per-halo object: no point-point pair, and no object in
    both for the shot noise.  With it the four branch pairs add to the unsplit
    auto-spectrum exactly.

    This is the one place that may know it, because it is the one place that
    knows sectors.  A caller's ``overlaps`` entry still wins.
    """
    if not (wa.discrete and wb.discrete) or ca.sector != cb.sector:
        return None
    if ca.sector != "agn" or ca.band != cb.band:
        return None
    if {ca.view, cb.view} != set(_AGN_BRANCH_VIEWS):
        return None
    return Overlap(point="same_object", shared="none",
                   why="an AGN is obscured or it is not: the two branch views "
                       "partition one population, so a halo's central AGN is "
                       "in one of them and never both")


def _check_gas_closure(spec, field, sectors, params):
    r"""Warn when a composite counts the gas of two models.

    A tracer holding both a ``gas`` mass-like view and ``ejecta`` -- the
    ``electrons`` composite is the shipped one -- is only a budget if the
    split's hot fraction is the DPM's.  Checked on the :math:`k\to0` weights,
    and only outside a trace, where the values are concrete.  Reported rather
    than refused: a caller may pair a fitted split with the DPM on purpose, and
    :meth:`~ggah_mod.sectors.ejecta.EjectaSector.split_from_gas` is the way to
    make them one model.
    """
    import warnings
    import jax
    comps = spec.components
    gas = [c for c in comps if c.sector == "gas" and c.view in ("density", "mass")]
    ej = [c for c in comps if c.sector == "ejecta"]
    if not (gas and ej):
        return
    w_gas = build_weights(gas[0], field, sectors, params)
    w_ej = build_weights(ej[0], field, sectors, params)
    held = (w_gas.at_large_scales() + w_ej.at_large_scales()) / field.m
    ej_params = params["ejecta"]
    split = ej_params.get("split") if isinstance(ej_params, dict) else None
    if split is None:
        return
    want = jnp.atleast_1d(split.f_hot) + jnp.atleast_1d(split.f_ejected)
    miss = jnp.max(jnp.abs(held - want))
    # Any traced input -- a parameter under grad, a field under jit -- makes
    # this a tracer, and a tracer has no value to compare.
    if isinstance(miss, jax.core.Tracer):
        return
    miss = float(miss)
    if miss > GAS_CLOSURE_TOL:
        warnings.warn(
            f"tracer {spec.name!r} adds the DPM's gas to the split's ejected "
            f"gas, and the two do not close on the split's non-stellar "
            f"baryons: they miss by up to {miss:.2e} of the halo mass.  The "
            f"split's hot fraction is not the DPM's, so some gas is counted "
            f"twice or not at all.  Build the split with "
            f"EjectaSector.split_from_gas to make them one model.",
            RuntimeWarning, stacklevel=3)


def matter_suppression(field, sectors: dict, params: dict, *,
                       options: PkOptions = PkOptions(), bnl_table=None,
                       transition_params=None):
    r""":math:`S(k) = P_{mm}/P_{mm}^{\rm DMO}`, the baryonic suppression.

    The number weak lensing surveys have to marginalise over, and the one this
    whole thread changes: before the six-way split, the baryons a halo had lost
    were riding :math:`\tilde u_{\rm DM}`, so the model could not produce a
    suppression at all -- moving mass from one component to another with the
    same profile is the identity.

    The dark-matter-only reference is built by **dropping the profiles, not the
    fractions**.  Every component keeps its mass and is put back on
    :math:`\tilde u_{\rm DM}`, which is exactly what
    :func:`~ggah_mod.sectors.matter.matter_weights` does when no profile is
    passed.  So the two spectra differ in *where* the baryons are and in
    nothing else -- not in how many there are, not in the halo mass function,
    not in the counterterm.  A reference built by zeroing the baryons instead
    would fold the change in :math:`\bar\rho` into the ratio and report a
    suppression that is partly a different cosmology.

    Returns
    -------
    (k, S) : tuple of arrays
        ``S -> 1`` at small :math:`k`, where every :math:`\tilde u \to 1` and
        the profiles cannot be distinguished.  That limit is a check, not a
        boundary condition: nothing here imposes it.
    """
    if "matter" not in sectors or "matter" not in params:
        raise ValueError(
            "matter_suppression needs the matter sector and its parameters; "
            "it is a ratio of two matter spectra and there is nothing to take "
            "a ratio of without them")
    if options.bnl and bnl_table is None:
        from .bnl import table_for
        bnl_table = table_for(field, options)
    p_bary = spectrum(field, "matter", "matter", sectors, params,
                      options=options, bnl_table=bnl_table,
                      transition_params=transition_params)
    dmo_params = dict(params)
    dmo_params["matter"] = {"split": params["matter"]["split"]}
    # The neutral gas's profile arrives as a peer rather than in the matter
    # block, so dropping the block's transforms is not enough to strip it: the
    # reference is every component on u_DM, and the cold gas is a component.
    dmo_params.pop("coldgas", None)
    p_dmo = spectrum(field, "matter", "matter", sectors, dmo_params,
                     options=options, bnl_table=bnl_table,
                     transition_params=transition_params)
    return p_bary.k, p_bary.total / p_dmo.total
