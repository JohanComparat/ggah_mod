r"""What a tracer *is*, declared statically.

Layer 4's integral takes a pair of tracers.  Two facts about that pair cannot be
read off :class:`~ggah_mod.sectors.protocol.TracerWeights`, and both change the
answer:

**A tracer may be more than one component.**  The X-ray field is the hot gas
(a continuous emissivity) *plus* the AGN (a discrete point population).  They
cannot be one ``TracerWeights``: :func:`~ggah_mod.sectors.protocol.combine`
refuses to add a discrete tracer, and it is right to, because a self-pair rule
between two populations is not something a sum can carry.  So

.. math::

    P_{XX} = P_{\rm gas,gas} + 2P_{\rm gas,agn} + P_{\rm agn,agn}

is what the predecessor's ``_pk_tables_XX`` writes out by hand.  Here it is
:func:`~ggah_mod.spectra.tracers.spectrum` looping over the *ordered* component
pairs: four :func:`~ggah_mod.spectra.pk.pk_cross` evaluations -- the cross pair
once in each order -- under two rules: the full product for the continuous gas
auto and for both mixed pairs, and the discrete self-pair rule for the AGN
auto, which pairs no central nucleus with itself and, for centrals-only AGN
(``f_duty_sat = 0``), leaves no one-halo pair term at all.  "One integral for any tracer pair" is true once a
tracer is a *list of components*, and false before.

**Whether two discrete tracers share objects.**  The full product
:math:`(w_a)(w_b)` is right only when no object is in both samples.  An X-ray
AGN sits in a galaxy, and in a magnitude-limited sample that galaxy is usually
*in* the galaxy sample -- so :math:`\langle N_c^g N_c^A\rangle = \langle
N_c^A\rangle`, not the product, and the difference is :math:`1/\bar n_g`.  Flat
in k, and invisible in any shape comparison.

The predecessor made both choices, in two files, and recorded neither: its
``_pk_tables_XX`` excludes the AGN central self-pair with the comment "no
central self-pair", while ``forward_jax``'s ``Xtot = X + Xa`` squares it with the
self-pair left in.  Two implementations of one spectrum, disagreeing at the
:math:`1/N_c` level.

Hence :attr:`Component.population`: an object-identity label, empty for a
continuous field.  Crossing two discrete components that do not declare their
relationship **raises**.  That is the same move :mod:`~ggah_mod.sectors.agn`
makes for the halo-abundance-matching model and
:func:`~ggah_mod.sectors.cooling.make_cooling` makes for a missing table --
refuse, and say what is missing.

Everything here is frozen, hashable and free of arrays: these objects live in a
``jax`` treedef, never among its leaves.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Overlap", "Band", "Component", "TracerSpec", "PkOptions", "OVERLAPS",
           "overlap_of"]

#: How two tracers' object samples relate.
#:
#: ``"none"``       -- disjoint; no object is in both.  Full product.
#: ``"identical"``  -- the same objects.  The auto-spectrum rule.
#: ``"nested"``     -- every object of one is in the other.  Declared, and
#:                     refused: see :func:`overlap_of`.
OVERLAPS = ("none", "identical", "nested")


@dataclass(frozen=True)
class Overlap:
    r"""The per-component-pair rule two discrete tracers cannot infer.

    :func:`overlap_of` answers from ``population`` alone, and for a *nested*
    pair it cannot: the label says the samples overlap, not *which components*
    are shared.  The AGN centrals may be a subset of the galaxy centrals while
    the satellites are unrelated, and the rule differs per component pair.  This
    is that statement, declared by the caller and carried into the integrand.

    It is the value type of ``spectrum(..., overlaps=...)``, whose keys are
    already per-``Component`` population labels.  What was missing was never the
    key -- it was a value meaning *"these two point populations are the same
    one-per-halo object"*.

    **Why one bit and not a matrix.**  Enumerate the 2x2 of point against
    extended: a point object is never an extended one (a central is not a
    satellite), so the cross terms always stand; satellites are Poisson about
    the central, so :math:`\langle N(N-1)\rangle = \langle N\rangle^2` and the
    extended-extended term stands too.  The *only* coefficient that is ever
    zero is point-point, and it is zero exactly when the two point populations
    are drawn from the same one-per-halo object.  ``"identical"`` on a discrete
    tracer is already that corner; this is the same corner for two different
    samples.

    Attributes
    ----------
    point : {"disjoint", "same_object"}
        Whether the two tracers' **point** populations are the same one-per-halo
        object.  ``"same_object"`` drops :math:`p_ap_b` from the one-halo pair
        **entirely** -- not :math:`\langle N_aN_b\rangle - \langle N_{a\cap b}\rangle`,
        which is a different and wrong subtraction: a halo has one central, so
        the whole product *is* the same-object case and no distinct pair is
        left.
    shared : {"none", "point", "subset"}
        How many objects are in **both** samples, per halo, for the cross shot
        noise.  ``"point"`` is :math:`N^p_aN^p_b`, the two point selections
        independent at fixed halo mass -- which is a model, and is named here
        rather than assumed.  ``"subset"`` is
        :math:`\min(\langle N_a\rangle, \langle N_b\rangle)`, exact when one
        sample's objects are all in the other *and* both are number counts;
        for a luminosity-weighted population the shared term is
        :math:`\langle N\ell_a\ell_b\rangle` and no algebra on the two
        self-pairs recovers it.
    why : str
        Required, and for the reason a
        :class:`~ggah_mod.sectors.params.Param` requires one: an unstated model
        is a hard-coding with extra steps.
    """

    point: str = "disjoint"
    shared: str = "none"
    why: str = ""

    _POINT = ("disjoint", "same_object")
    _SHARED = ("none", "point", "subset")

    def __post_init__(self):
        if self.point not in self._POINT:
            raise ValueError(f"Overlap.point must be one of {self._POINT}, "
                             f"got {self.point!r}")
        if self.shared not in self._SHARED:
            raise ValueError(f"Overlap.shared must be one of {self._SHARED}, "
                             f"got {self.shared!r}")
        if not str(self.why).strip():
            raise ValueError(
                "an Overlap needs a `why`.  It states a model -- which objects "
                "two samples share -- that the package refused to guess, and a "
                "declaration nobody has to justify is the guess with an extra "
                "step.  Say what makes the two point populations the same "
                "object, or disjoint.")


@dataclass(frozen=True)
class Band:
    """An energy band, in keV.

    ``frame`` is explicit because the two answers differ by :math:`(1+z)` and
    neither package that came before ever handled the observer frame at all --
    both build their APEC tables at ``redshift=0``.
    """

    emin: float
    emax: float
    frame: str = "observer"

    def __post_init__(self):
        if not 0.0 < self.emin < self.emax:
            raise ValueError(f"a band needs 0 < emin < emax, got "
                             f"({self.emin}, {self.emax})")
        if self.frame not in ("observer", "rest"):
            raise ValueError(f"unknown band frame {self.frame!r}; expected "
                             f"'observer' or 'rest'")

    def label(self) -> str:
        return f"{self.emin:g}-{self.emax:g}keV/{self.frame[0]}"


@dataclass(frozen=True)
class Component:
    """One sector's contribution to a tracer.

    Attributes
    ----------
    sector : str
        Which layer-3 sector answers: ``matter``, ``galaxies``, ``gas``, ``agn``.
    view : str
        Which of that sector's views -- ``cen``/``sat``/``total`` for galaxies,
        ``mass``/``density``/``pressure``/``xray`` for gas, ``counts``/
        ``emission`` for AGN.
    population : str
        **The object-identity label.**  Empty for a continuous field, where the
        question does not arise.  Two discrete components with the same
        non-empty label are the *same physical objects*; with different labels
        they are disjoint samples.  A discrete component with no label cannot be
        crossed with another discrete component -- see :func:`overlap_of`.
    band : Band or None
        For the X-ray views.
    """

    sector: str
    view: str = ""
    population: str = ""
    band: Band | None = None

    def label(self) -> str:
        """Short name, for a parity-budget row or an error message."""
        parts = self.sector if not self.view else f"{self.sector}:{self.view}"
        return parts if self.band is None else f"{parts}[{self.band.label()}]"


@dataclass(frozen=True)
class TracerSpec:
    """A named tracer: one or more :class:`Component`."""

    name: str
    components: tuple[Component, ...]

    def __post_init__(self):
        if not self.components:
            raise ValueError(f"tracer {self.name!r} has no components")

    @classmethod
    def one(cls, sector: str, view: str = "", *, population: str = "",
            band: Band | None = None, name: str | None = None) -> "TracerSpec":
        """A single-component tracer, which is most of them."""
        c = Component(sector, view, population, band)
        return cls(name or c.label(), (c,))

    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class PkOptions:
    """Static choices for one spectrum.  No arrays: this is treedef material.

    Attributes
    ----------
    two_halo_spectrum : str
        ``"cb"`` or ``"total"``.  See :func:`~ggah_mod.spectra.pk.pk_2h`.
    two_halo_consistency : str
        ``"linear_deficit"`` or ``"none"``.  See
        :func:`~ggah_mod.spectra.pk.low_mass_counterterm`.
    bnl : bool
        Apply the beyond-linear-bias correction to the two-halo term, with its
        low-mass completion (:mod:`~ggah_mod.spectra.bnl`).  On by default.
    one_halo_transition : str
        Which one-halo/two-halo transition multiplies :math:`P^{1h}`.  See
        :mod:`~ggah_mod.spectra.transition`.  ``"none"``, the literal sum, is
        the default; ``"mead20"`` is HMcode-2020's damping, for Fourier-space
        matter spectra.
    neutrino_two_halo : str
        ``"linear"`` adds the neutrinos' linear leg to the two-halo amplitude of
        every tracer that carries them (:mod:`~ggah_mod.spectra.neutrinos`);
        ``"none"`` keeps the cold-only halo model.  Refused with
        ``two_halo_spectrum="total"``, which already contains them.
    """

    two_halo_spectrum: str = "cb"
    two_halo_consistency: str = "linear_deficit"
    bnl: bool = True
    one_halo_transition: str = "none"
    neutrino_two_halo: str = "linear"

    def __post_init__(self):
        if self.two_halo_spectrum == "total" and self.neutrino_two_halo == "linear":
            raise ValueError(
                "two_halo_spectrum='total' with neutrino_two_halo='linear' "
                "counts the neutrinos twice: the total spectrum already "
                "contains them, and the leg adds them again.  Pair 'total' "
                "with neutrino_two_halo='none'.")

    @classmethod
    def from_backend(cls, backend, **overrides) -> "PkOptions":
        """Take the flavour's declared choices, with explicit overrides.

        The declared set is built first and the overrides applied on top, so
        an override of a field the backend declares replaces it -- it used to
        be passed twice and raise ``TypeError``.
        """
        declared = dict(two_halo_spectrum=backend.two_halo_spectrum,
                        two_halo_consistency=backend.two_halo_consistency,
                        bnl=backend.bnl,
                        one_halo_transition=backend.one_halo_transition,
                        neutrino_two_halo=backend.neutrino_two_halo)
        return cls(**{**declared, **overrides})


def overlap_of(a: Component, b: Component, *, a_discrete: bool,
               b_discrete: bool) -> str:
    """How ``a``'s and ``b``'s object samples relate, or a refusal.

    A continuous field has no objects and therefore no self-pairs, so any pair
    involving one is ``"none"`` -- the full product -- whatever the labels say.

    For two discrete components the **label** decides, and it is read as a
    ``/``-separated path.  That is what lets one scheme answer three different
    questions without layer 4 knowing anything about galaxies or AGN:

    ==========================  ==========================  =============
    ``population`` of a         ``population`` of b         result
    ==========================  ==========================  =============
    ``galaxies/cen``            ``galaxies/cen``            identical
    ``galaxies/cen``            ``galaxies/sat``            none
    ``galaxies``                ``galaxies/cen``            nested
    ``galaxies``                ``agn``                     none
    ``galaxies``                *(empty)*                   **raises**
    ==========================  ==========================  =============

    Equal paths are the same objects.  **Siblings** -- neither a prefix of the
    other -- are disjoint samples, which is a *declaration by the labeller*, not
    a guess: two different top-level names mean two different samples, and a
    caller who means the AGN to sit inside the galaxy sample says so by calling
    it ``galaxies/agn``.  A **prefix** is the nested case, and
    :func:`~ggah_mod.spectra.pair.pair_1h` refuses it rather than inventing a
    rule.

    An unlabelled discrete component is a question, never a default.
    """
    if not (a_discrete and b_discrete):
        return "none"
    if a == b:
        return "identical"

    missing = [c.label() for c in (a, b) if not c.population]
    if missing:
        raise ValueError(
            f"cannot cross the discrete tracers {a.label()!r} and "
            f"{b.label()!r}: {missing} declare no `population`, so there is no "
            f"way to tell whether an object counted in one is also counted in "
            f"the other.  It is not a detail -- if they overlap, the one-halo "
            f"term must drop the shared objects' self-pairs, and the answer "
            f"differs by 1/n_bar at every k.  An X-ray AGN sits in a galaxy, "
            f"and in a magnitude-limited sample that galaxy is usually in the "
            f"galaxy sample too.  Give both components a `population` path: "
            f"the same string if they are the same objects, sibling names "
            f"('galaxies/cen', 'galaxies/sat') if the samples are disjoint, "
            f"and a prefix ('galaxies' against 'galaxies/agn') if one is "
            f"contained in the other.")

    pa, pb = a.population, b.population
    if pa == pb:
        return "identical"
    if _is_prefix(pa, pb) or _is_prefix(pb, pa):
        return "nested"
    return "none"


def _is_prefix(short: str, long: str) -> bool:
    """Is ``short`` a proper path-prefix of ``long``?  ``a`` of ``a/b``, yes."""
    return long.startswith(short + "/")
