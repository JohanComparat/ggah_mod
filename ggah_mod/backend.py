r"""The two pipeline flavours, as one declarative object.

A halo-model package always ends up with two ways to evaluate the same forward
model: an accurate one for fitting, and a fast differentiable one for
forecasting.  The failure mode is letting them become two *implementations*.
They then drift, and nothing records which one produced a number.

A :class:`Backend` is the alternative: one object carrying every knob on which
the two flavours legitimately differ -- the linear-P(k) solver, the growth
route, the grid resolutions, the quadrature node counts, the Hankel engine,
the concentration relation -- and one flag, :attr:`Backend.traced`, saying whether the assembly it
drives may go through ``jax.jit`` / ``jax.grad`` end to end.

The mechanism that lets one assembly serve both: the P(k) solver is called once,
up front, and its output is carried downstream as arrays rather than as a
callable.  Under ``ACCURATE`` those are concrete float64 arrays and everything
downstream evaluates eagerly; under ``DIFFERENTIABLE`` they are tracers and the
whole thing composes under ``jit``.  Downstream there is one code path written in
``jnp``, not a twin pair.

The object that carries them is :class:`~ggah_mod.halos.field.HaloField`, and it
is worth saying that this docstring named a ``cosmology.snapshot.CosmologySnapshot``
for a long time and no such module was ever written.  The design landed in layer 2
rather than layer 1 for a reason: the spectrum has to be snapshotted together with
the mass grid, the variance and the tables derived from both, and splitting them
across two objects would let a field hold :math:`\sigma(M)` from one spectrum and
``pk_cb`` from another.  ``HaloField`` holds ``k``, ``pk_cb`` and ``pk_lin``
alongside them, at one redshift, and ``make_fields`` is what turns a redshift set
into a tuple of them in a single solve.

``traced`` is validated, not asserted.  The dangerous direction is a backend
that *claims* to be differentiable while some part of it is not: the gradient
then succeeds and is quietly missing a term.  ``ACCURATE.with_(traced=True)``
raises for exactly that reason -- a gradient through a Boltzmann solver would
run and omit the dependence of P(k) on the cosmology.

What choosing ``DIFFERENTIABLE`` costs in accuracy is deliberately not stated here.  It is
measured, row by row, in the parity budget.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace

__all__ = ["Backend", "ACCURATE", "DIFFERENTIABLE", "DIFFERENTIABLE_COARSE",
           "BACKENDS", "resolve_backend", "ENV_VAR", "PK_ENV_VAR",
           "HANKEL_ENGINES", "TWO_HALO_SPECTRA", "TWO_HALO_CONSISTENCY"]


@dataclass(frozen=True)
class Backend:
    """A pipeline flavour.  Frozen: caches are keyed on it."""

    name: str
    #: May the assembly driven by this backend be jit/grad-ed end to end?
    traced: bool

    # -- layer 1 ------------------------------------------------------------
    #: Linear P(k) backend key, see :mod:`ggah_mod.cosmology.power`.
    pk: str

    # -- grids --------------------------------------------------------------
    n_k: int
    k_min: float
    k_max: float
    n_m: int
    m_min: float
    m_max: float

    # -- layer 4 ------------------------------------------------------------
    #: Which linear spectrum multiplies the two-halo term: ``"cb"`` (cold) or
    #: ``"total"``.  ``sigma(M)``, ``b(M)`` and ``dn/dM`` are all built from
    #: ``pk_cb``, so the peak-background split that defines ``b(M)`` is a
    #: statement about the cold field; ``"cb"`` is the default for that reason.
    #: The disagreement is a parity-budget row, not a rounding choice.
    two_halo_spectrum: str
    #: How the two-halo normalisation deficit is handled.  Over the shipped
    #: mass range ``int b (M/rho_cb) dn/dM dM`` is **0.70**, not 1, so an
    #: uncorrected ``P_mm^2h`` is short by a factor of two at large scales.
    #:
    #: ``"linear_deficit"`` adds ``1 - int b (M/rho_cb) dn/dM dM`` as a
    #: ``b = 1``, ``u = 1`` component; ``"none"`` leaves it, and reports.
    #:
    #: The quantity is named rather than described, because "the unresolved
    #: mass" reads two ways and one of them is wrong by 40 per cent.  The
    #: missing *mass fraction* is a different number -- ``1 - int (M/rho_cb)
    #: dn/dM dM`` = 0.48 against the bias deficit's 0.30 -- and adding that
    #: instead overshoots the k -> 0 limit, because the unresolved mass is not
    #: unbiased: it lies below 1e10 Msun/h where ``b < 1``, and the effective
    #: bias the budget requires is 0.62, not the ``b = 1`` this adds.  The two
    #: coincide only if it were.  Found by ``ggah_mod_benchmark`` implementing
    #: the older wording literally and measuring 40 per cent against CCL.
    two_halo_consistency: str
    #: Whether the two-halo term carries the beyond-linear halo bias of Mead &
    #: Verde (2021), with its low-mass completion.  ``P_lin I_a I_b`` is 10-50
    #: per cent short across the one-halo/two-halo transition, and the measured
    #: correction closes most of it: against CLASS's HMcode-2020 the matter
    #: spectrum goes from 11.8 to 3.2 per cent rms over 0.05 < k < 1 at
    #: z = 0.5.  A statement about the model, so both flavours declare it.
    #: See ``ggah_mod.spectra.bnl``.
    bnl: bool
    #: Which one-halo/two-halo transition multiplies ``P_1h``.
    #:
    #: ``"none"``, the literal sum, is the default: ``xi_1h`` has compact
    #: support, so the ``k -> 0`` plateau of ``P_1h`` does not reach ``xi``,
    #: ``w_p`` or ``Delta Sigma`` at ``r > 0``, it is physical Poisson power for
    #: rare-halo tracers, and the transition region is repaired in the two-halo
    #: term by ``bnl``.  ``"mead20"`` is HMcode-2020's damping,
    #: ``(k/k_*)^4/(1+(k/k_*)^4)`` with ``k_* = 0.05618 sigma8_cb^-1.013``, for
    #: Fourier-space matter spectra at ``k < 0.05``, where mass conservation
    #: wants ``k^4``.  ``"mead15"`` is retired: it paired the 2020 form with the
    #: 2015 scale.  See ``ggah_mod.spectra.transition``.
    #:
    #: Both flavours declare the same value: this is a statement about the
    #: model, not about how fast it has to run, so a flavour that disagreed here
    #: would make the parity budget measure a physics difference and report it
    #: as a numerical one.
    one_halo_transition: str
    #: How the massive neutrinos enter the two-halo term.  They are in no halo,
    #: so the halo integral cannot carry them; ``"linear"`` adds their linear
    #: leg :math:`\sqrt{P_m/P_{cb}} - \bar\rho_{cb}/\bar\rho_m` to the
    #: amplitude of every tracer that carries them, which makes
    #: ``P_mm^2h -> P_m`` on large scales rather than ``(1-f_nu)^2 P_cb``.
    #: ``"none"`` is the cold-only halo model.  ``"total"`` with ``"linear"``
    #: is refused: that spectrum already contains the neutrinos.  See
    #: ``ggah_mod.spectra.neutrinos``.
    neutrino_two_halo: str

    # -- layer 5 quadratures ------------------------------------------------
    n_chi: int
    #: **A line-of-sight limit for Sigma and Delta Sigma, in Mpc/h.  Not a
    #: Limber grid** -- which the name now says, rather than relying on this
    #: paragraph being read.  It was ``chi_max``, and a field called that in a
    #: package with a Limber projection reads as the projection's ceiling; the
    #: docstring said otherwise and a docstring is not where a reader looks
    #: first.  ``PLAN.md`` item **D3**.
    #:
    #: It, ``n_chi`` and ``n_r_tab`` were copied together from the
    #: predecessor's ``_delta_sigma_from_pgm(chi_max=300., n_chi=512,
    #: n_r_tab=256)``, where 300 Mpc/h is generous: a halo profile truncated at
    #: a few Mpc has nothing left out there.  A **Limber** integral is a
    #: different quantity and needs chi out to z ~ 3, about 4700 Mpc/h; used as
    #: it stands it would truncate every C_ell at z ~ 0.075.  So layer 5's
    #: projection derives its own chi grid from the spec's n(z) support and
    #: does not read this.
    chi_max_los: float
    #: Line-of-sight nodes of ``w_p``, per :math:`r_p`, uniform in
    #: :math:`{\rm asinh}(\pi/r_p)` (``observables.real_space._los_integral``).
    #: Until 0.9.7 they were uniform in :math:`\pi`, and too coarse below
    #: :math:`r_p \approx 0.2` Mpc/h.
    n_pi: int
    #: Points of the :math:`\xi` table ``w_p`` reads, log-spaced over
    #: :math:`10^{-3}`-:math:`10^{2.5}` Mpc/h.
    n_r_tab: int
    #: How far FFTLog extends the input grid at **each** end, continuing the
    #: function by the same capped power law :func:`~ggah_mod.observables.
    #: transforms.hankel` uses past its own grid, expressed in decades of x.
    #:
    #: FFTLog treats its input as periodic in ``ln x``.  When the integrand has
    #: not decayed at the edge -- and for a galaxy auto-spectrum it has not:
    #: at ``k_max = 200`` the one-halo term leaves the log-slope at ``-1.79``,
    #: so ``k^3 P`` is still at its maximum -- the wrap is a step, and it
    #: rings.  Measured against the Ogata flavour on the same spectrum, inside
    #: 2 Mpc/h: **2.8e-1 unpadded, 1.0e-3 at two decades**.
    #:
    #: This is not a resolution knob and widening ``n_k`` does not substitute
    #: for it: FFTLog is stable to four digits across ``n_k`` 512 to 4096 on
    #: the same input.  The quadrature engine needs none of this because it
    #: never wraps -- it interpolates with a tail instead.
    #:
    #: Stated in **decades**, not nodes, so it means the same thing on any
    #: grid: a node count would make the physical reach depend on ``n_k``,
    #: which is the resolution knob this one is explicitly not.
    #:
    #: ``0.0`` restores the unpadded behaviour, which is what the parity
    #: budget's small-scale row was measuring.
    fftlog_pad_decades: float
    #: Ogata's node count -- and **read by one flavour of the three**.
    #:
    #: ``ACCURATE`` declares ``hankel="quadrature"`` and reads it;
    #: ``DIFFERENTIABLE`` and ``DIFFERENTIABLE_COARSE`` declare ``"fftlog"``,
    #: whose length is the ``k`` grid's, and carry a value that does nothing.
    #: Same for :attr:`hankel_h`.
    #:
    #: The package's "every ``Backend`` field has a reader" audit passes,
    #: because *some* flavour reads them -- which is the gap: readership is per
    #: flavour and the audit is global.  Recorded rather than removed, because
    #: a flavour that switched engine would want them, and a field deleted for
    #: being unread on two of three is a field that has to be reinvented.
    #: ``PLAN.md`` item **D5**.
    n_hankel: int
    #: Redshift nodes in a Limber projection.
    n_z_proj: int
    #: ``"quadrature"`` (Ogata double-exponential) or ``"fftlog"``.  Both are
    #: pure JAX; the choice is accuracy against cost, and it is measured.
    hankel: str
    #: Ogata's step.  **This is the accuracy knob, and ``n_hankel`` is not.**
    #:
    #: Measured against a closed-form Gaussian transform pair, the error depends
    #: on ``hankel_h`` alone and is flat in ``n_hankel``:
    #:
    #: ====== ============ ============
    #: ``h``  xi rel. err  DS rel. err
    #: ====== ============ ============
    #: 0.020  4.4e-2       5.0e-2
    #: 0.010  8.4e-4       5.8e-3
    #: 0.005  2.1e-5       1.3e-4
    #: 0.002  7.8e-7       1.1e-8
    #: 0.001  1.0e-7       1.5e-8
    #: ====== ============ ============
    #:
    #: ``n_hankel`` sets the *reach* instead -- ``x_max``, and hence how far past
    #: the k grid the rule samples -- and saturates once ``h*N`` exceeds ~2.5.
    #: The predecessor used ``N = 512, h = 0.005`` and sits at 2e-5; the same
    #: 512 nodes at ``h = 0.001`` reach 1e-7.  Two knobs that look like one.
    #:
    #: **That table is conditional on the reach being adequate, and the
    #: condition is now enforced rather than written down.**  It was measured
    #: against a Gaussian, whose transform is a Gaussian: no oscillatory tail to
    #: cancel, so truncating ``x_max`` costs nothing and the step is genuinely
    #: the only knob.  A halo-model spectrum is not that.  ``xi(r)`` at large
    #: separation is a small residue of a strongly oscillatory integral, and
    #: stopping the rule early leaves an uncancelled tail comparable to the
    #: answer.  Measured on a ``zheng07`` galaxy spectrum at PLANCK18, z = 0,
    #: against the converged value:
    #:
    #: ======  ======  ======  =============  ==============
    #: ``h``   ``N``   ``hN``  xi err r = 10  xi err r = 50
    #: ======  ======  ======  =============  ==============
    #: 0.0010  512     0.51    7.5e-2         2.7e-1
    #: 0.0010  1024    1.02    3.3e-2         2.8e-1
    #: 0.0010  2048    2.05    8.3e-4         1.6e-2
    #: 0.0050  512     2.56    1.1e-3         1.8e-3
    #: 0.0010  4096    4.10    2.4e-4         2.1e-3
    #: ======  ======  ======  =============  ==============
    #:
    #: ``ACCURATE`` shipped ``h = 0.001, N = 512`` -- ``hN = 0.51``, the first
    #: row -- so the flavour named for its accuracy was **27 per cent** low in
    #: ``xi`` at 50 Mpc/h while the differentiable one was right.  The threshold
    #: was in this docstring the whole time and nothing read it, which is the
    #: same shape of defect as a ``Backend`` field with no reader.  ``_validate``
    #: reads it now.
    hankel_h: float

    # -- layer 3 ------------------------------------------------------------
    n_gl: int
    #: Concentration relation; must be traceable when :attr:`traced`.
    cm_model: str
    #: Halo mass definition.
    mdef: str
    #: Multiplicity function.
    #:
    #: Here, rather than only as a ``make_field`` default, because the four
    #: halo choices are one choice and a flavour that declared two of them was
    #: declaring half of it.  With all four in one object :func:`_validate` can
    #: run the calibration check at construction -- before any cosmology
    #: exists -- so an inconsistent flavour cannot be built at all.
    hmf_model: str = "tinker08_csst"
    #: Linear bias.  Should be the peak-background-split partner of
    #: :attr:`hmf_model`; ``_validate`` says so if it is not.
    bias_model: str = "tinker10"

    def with_(self, **overrides) -> "Backend":
        """A copy with fields replaced, re-validated."""
        return _validate(replace(self, **overrides))

    def __str__(self) -> str:  # pragma: no cover
        return (f"{self.name}(pk={self.pk}, n_k={self.n_k}, n_m={self.n_m}, "
                f"n_gl={self.n_gl}, mdef={self.mdef}, "
                f"hmf={self.hmf_model}, bias={self.bias_model}, "
                f"cm={self.cm_model}, traced={self.traced})")


#: P(k) backends that are pure JAX.  One, since CosmoPower-JAX left.
TRACEABLE_PK = frozenset({"emu_pk"})
#: **Removed: ``gas_ft``.**  It offered ``"quadrature"`` and ``"surrogate"``,
#: the traced flavour (then named ``FAST``) declared the second, and
#: ``_validate`` refused anything else on a
#: traced backend on the grounds that the first was "the numpy quadrature
#: path".
#: Every part of that was false:
#:
#: * :func:`~ggah_mod.halos.profiles.profile_uk_gl` is jnp throughout and
#:   differentiates, so neither value was ever excluded from a traced backend;
#: * nothing implemented ``"surrogate"`` -- :class:`~ggah_mod.sectors.gas.HotGasDPM`
#:   took the quadrature path whatever the field said;
#: * nothing *could* implement it: the DPM shape is a gNFW with free slopes,
#:   whose transform has no closed form, so :func:`~ggah_mod.halos.profiles.gnfw_uk`
#:   is itself the same quadrature.
#:
#: A field with one legal value declares no choice, so it is gone rather than
#: kept as a knob that cannot turn.  The accuracy difference between the two
#: flavours is :attr:`Backend.n_gl` -- 200 against 64 -- which is now read.
#: ``tests/test_backends.py::TestEveryBackendFieldHasAReader`` is what found it,
#: and is what stops the next one.

#: Hankel transform engines, layer 5.
HANKEL_ENGINES = frozenset({"quadrature", "fftlog"})

#: Which linear spectrum multiplies the two-halo term.
TWO_HALO_SPECTRA = frozenset({"cb", "total"})

#: What to do about the two-halo normalisation deficit.
TWO_HALO_CONSISTENCY = frozenset({"linear_deficit", "none"})

#: The smallest ``hankel_h * n_hankel`` an Ogata rule may declare.
#:
#: Not a tolerance but a cliff.  Above ~2.5 the rule has saturated and further
#: reach changes nothing; below it the transform is truncated before its
#: oscillatory tail has cancelled, and the error at large separation is tens of
#: per cent rather than the 1e-7 the step alone would suggest.  See
#: :attr:`Backend.hankel_h` for the measured table.
MIN_OGATA_REACH = 2.5

#: One-halo/two-halo transitions.  Kept in step with
#: ``spectra.transition.ONE_HALO_TRANSITION`` by a test rather than by an
#: import: layer 4 may read this module, and this module reading layer 4 back
#: would be the cycle the coherence audit exists to forbid.
ONE_HALO_TRANSITIONS = frozenset({"none", "mead20"})

#: How the neutrinos enter the two-halo term.  Kept in step with
#: ``spectra.neutrinos.NEUTRINO_TWO_HALO`` by a test, for the same reason.
NEUTRINO_TWO_HALO = frozenset({"linear", "none"})

#: Concentration relations that are pure JAX.
#:
#: All six of them are, because this package implements them natively rather
#: than calling colossus -- which is what the predecessor did, and which is
#: numpy *and* mutates global state through ``setCosmology``.  So the set is
#: currently the whole registry, and it is still written out rather than derived
#: from it: a relation added later that wraps an external library must be left
#: out of here deliberately, and a set that derived itself would silently
#: include it.
#:
#: ``tests/test_backends.py`` checks both directions -- that every name here
#: exists in the registry, and that every member really does admit a gradient.
TRACEABLE_CM = frozenset({
    "duffy08", "dutton14", "klypin16", "bhattacharya13", "diemer19", "seppi21",
})


def _validate(b: Backend) -> Backend:
    if b.traced:
        for value, allowed, what in (
            (b.pk, TRACEABLE_PK, "linear P(k) backend"),
            (b.cm_model, TRACEABLE_CM, "concentration relation"),
        ):
            if value not in allowed:
                raise ValueError(
                    f"backend {b.name!r} is traced but its {what} {value!r} is "
                    f"not differentiable; use one of {sorted(allowed)}")
    # The declared names have to resolve.  Structural checks below catch an
    # empty grid; nothing caught a *name*, and two were wrong: ACCURATE asked
    # for "diemer19" and TRACEABLE_CM offered "diemer19_jax", neither of which
    # is in the registry (the relation is called "diemer19", after the paper it
    # first appeared in).  Both went unnoticed because no layer consumes
    # ``cm_model`` yet -- layer 3 would have raised on its first call.
    #
    # Checked for every backend, not only traced ones: ACCURATE is the one that
    # was wrong, and ``traced`` is exactly the branch it never enters.
    #
    # Imported here rather than at module scope so that importing the flavour
    # selector does not drag in layer 2 before anything asks for it.
    from .halos import CONCENTRATION
    from .halos.mass_definitions import parse_mass_def

    if b.cm_model not in CONCENTRATION:
        raise ValueError(
            f"backend {b.name!r} names a concentration relation "
            f"{b.cm_model!r} that does not exist; expected one of "
            f"{sorted(CONCENTRATION)}")
    try:
        parse_mass_def(b.mdef)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"backend {b.name!r} names a mass definition {b.mdef!r} that does "
            f"not parse: {exc}") from exc

    # The four together, checked once, at construction.  A flavour is a frozen
    # set of choices and this is the only moment at which all four are known
    # and none has been used yet; leaving it to `make_field` would mean the
    # error arrives per call, after a cosmology has been built for it.
    from .halos.calibration import check_calibration

    try:
        check_calibration(b.hmf_model, b.cm_model, b.mdef, 0.0,
                          policy="strict", bias_model=b.bias_model)
    except ValueError as exc:
        raise ValueError(
            f"backend {b.name!r} declares a set of halo choices that are not "
            f"calibrated together: {exc}") from exc

    for value, allowed, what in (
        (b.hankel, HANKEL_ENGINES, "Hankel engine"),
        (b.two_halo_spectrum, TWO_HALO_SPECTRA, "two-halo spectrum"),
        (b.two_halo_consistency, TWO_HALO_CONSISTENCY, "two-halo consistency"),
        (b.one_halo_transition, ONE_HALO_TRANSITIONS, "one-halo transition"),
        (b.neutrino_two_halo, NEUTRINO_TWO_HALO, "neutrino two-halo treatment"),
    ):
        if value not in allowed:
            raise ValueError(
                f"backend {b.name!r} names a {what} {value!r} that does not "
                f"exist; expected one of {sorted(allowed)}")

    if not isinstance(b.bnl, bool):
        raise ValueError(
            f"backend {b.name!r} declares bnl = {b.bnl!r}; it is a switch, "
            f"True or False")
    if b.two_halo_spectrum == "total" and b.neutrino_two_halo == "linear":
        raise ValueError(
            f"backend {b.name!r} pairs two_halo_spectrum='total' with "
            f"neutrino_two_halo='linear', which counts the neutrinos twice: "
            f"the total spectrum already contains them.  Declare "
            f"neutrino_two_halo='none' with it.")

    if b.hankel == "quadrature" and b.hankel_h * b.n_hankel < MIN_OGATA_REACH:
        raise ValueError(
            f"backend {b.name!r} pairs hankel_h = {b.hankel_h} with "
            f"n_hankel = {b.n_hankel}, so hankel_h * n_hankel = "
            f"{b.hankel_h * b.n_hankel:.3g} < {MIN_OGATA_REACH}.  Ogata's "
            f"reach saturates above that and is badly truncated below it: at "
            f"0.51 the correlation function is 27 per cent low at 50 Mpc/h, "
            f"which is not an accuracy setting but a wrong answer.  Raise "
            f"n_hankel, or raise hankel_h and accept the coarser step.")

    if not 0.0 < b.k_min < b.k_max:
        raise ValueError(f"backend {b.name!r} has an empty k range")
    if not 0.0 < b.m_min < b.m_max:
        raise ValueError(f"backend {b.name!r} has an empty mass range")
    for f in ("n_k", "n_m", "n_chi", "n_pi", "n_r_tab", "n_hankel", "n_gl",
              "n_z_proj"):
        if getattr(b, f) < 2:
            raise ValueError(f"backend {b.name!r} has {f} < 2")
    if not 0.0 < b.hankel_h < 0.1:
        raise ValueError(
            f"backend {b.name!r} has hankel_h = {b.hankel_h}, outside (0, 0.1). "
            f"Ogata's rule is unusable above ~0.02 (4% error) and the node "
            f"count needed to keep its reach grows as 1/h below ~1e-4.")
    return b


#: Fitting: a Boltzmann solver, fine grids, exact quadrature.
#:
#: The default P(k) backend is **CLASS**, not CAMB.  Both are Boltzmann solvers
#: and they agree to 0.15 percent on the shape, so the choice is not about
#: accuracy: CLASS costs 7.2 s against CAMB's 15.2 s for one cold solve at the
#: configuration this package actually runs, and it is the only backend that
#: treats the neutrino sector exactly rather than through a fitting form or a
#: table.  CAMB remains one ``with_`` away and is what arbitrates a CLASS result,
#: which is the job an independent implementation is for.
#:
#: **The definition is 200m, and the four halo choices are one choice.**  The
#: set is the one in which every rung can respond to a cosmology: SO 200m,
#: ``tinker08_csst`` (``tinker08`` recalibrated by ``emu_hmf``, which is fitted at
#: Rockstar 200m), ``tinker10`` for the bias because it is that fit's
#: peak-background-split partner, and ``bhattacharya13`` for the concentration,
#: which is calibrated at 200m as well as 200c.  ``diemer19`` is not in it and
#: cannot be: it is calibrated at 200c and nothing else, and
#: :mod:`ggah_mod.halos.calibration` refuses the pairing.  The flavour declared
#: 200c with ``diemer19`` until that set became the default.
ACCURATE = _validate(Backend(
    name="accurate", traced=False, pk="class",
    n_k=1024, k_min=1e-4, k_max=200.0,
    n_m=512, m_min=1e10, m_max=1e16,
    two_halo_spectrum="cb", two_halo_consistency="linear_deficit", bnl=True,
    one_halo_transition="none", neutrino_two_halo="linear",
    n_chi=512, chi_max_los=300.0, n_pi=512, n_r_tab=256, n_hankel=4096,
    fftlog_pad_decades=2.0,
    n_z_proj=64, hankel="quadrature", hankel_h=0.001,
    n_gl=200, cm_model="bhattacharya13", mdef="200m",
    hmf_model="tinker08_csst", bias_model="tinker10",
))

#: Forecasting: an emulator, coarser grids, differentiable end to end.
#:
#: Named for what it *is* rather than for how it feels.  "FAST" described a
#: consequence; the property a caller has to reason about is that this flavour
#: may be passed through ``jax.jit`` and ``jax.grad`` end to end, and the
#: accurate one may not.
#:
#: **The spectrum is ``emu_pk``, and now it is the only one.**  Against CLASS
#: at the fiducial, over :math:`10^{-3} < k < 10\,h\,{\rm Mpc^{-1}}`, it
#: reproduces the shape to 0.148%; over the whole band the variance integral
#: actually uses, out to :math:`k = 200`, to 0.16%.  That second number is the
#: one that decided this, and it is a property of the training set rather than
#: of the fit -- the network was trained to :math:`k = 200`, which is exactly
#: this flavour's ``k_max``.  Three things follow:
#:
#: * :math:`P_{cb}` is a trained output rather than :math:`P_m` times a
#:   distilled ratio, so the cold and total spectra cannot drift apart.  The
#:   correction of :mod:`emu_pk.ratio` is off this path entirely -- and, since
#:   CosmoPower-JAX left, off every path in this package.
#: * :math:`w_0` and :math:`w_a` are network inputs, so the dark-energy
#:   response costs what any other parameter costs instead of being read
#:   through four Hermite axes.
#: * the primordial power law is divided out of the training target and
#:   restored analytically, so ``dlnP/dn_s`` and ``dlnP/dln10A_s`` are exact to
#:   6e-8 and 2e-14 rather than to whatever a network happened to learn.
#:
#: What arbitrates an ``emu_pk`` result is now a Boltzmann solver rather than a
#: second network: ``ACCURATE`` on CLASS, one ``with_`` away.  That is a better
#: arbiter and a slower one, which is the trade a second implementation was
#: making implicitly anyway.
DIFFERENTIABLE = _validate(Backend(
    name="differentiable", traced=True, pk="emu_pk",
    n_k=512, k_min=1e-4, k_max=200.0,
    n_m=256, m_min=1e10, m_max=1e16,
    two_halo_spectrum="cb", two_halo_consistency="linear_deficit", bnl=True,
    one_halo_transition="none", neutrino_two_halo="linear",
    n_chi=256, chi_max_los=300.0, n_pi=256, n_r_tab=256, n_hankel=256,
    fftlog_pad_decades=2.0,
    n_z_proj=32, hankel="fftlog", hankel_h=0.005,
    n_gl=64, cm_model="bhattacharya13", mdef="200m",
    hmf_model="tinker08_csst", bias_model="tinker10",
))

#: The same path on coarser grids, for a forecast that is exploring rather than
#: reporting.  Every grid halves; nothing else changes, so a number computed
#: here and a number computed on :data:`DIFFERENTIABLE` differ by quadrature
#: alone and the difference is measurable rather than conceptual.
DIFFERENTIABLE_COARSE = _validate(DIFFERENTIABLE.with_(
    name="differentiable_coarse",
    n_k=256, n_m=128, n_chi=128, n_pi=128, n_r_tab=128, n_hankel=128,
    n_z_proj=16, n_gl=32,
))

BACKENDS = {b.name: b for b in
            (ACCURATE, DIFFERENTIABLE, DIFFERENTIABLE_COARSE)}

#: Which flavour drives the assembly.
ENV_VAR = "GGAH_BACKEND"
#: Which linear-P(k) backend that flavour uses.
#:
#: Two names because they are two choices, and conflating them is what made the
#: previous single name ambiguous: the flavour fixes the grids, the quadrature
#: orders and whether the assembly is traceable, while the P(k) backend is one
#: field *inside* it.  ``GGAH_BACKEND=accurate GGAH_PK_BACKEND=camb`` is the
#: second Boltzmann solver on the accurate path;
#: ``GGAH_BACKEND=differentiable GGAH_PK_BACKEND=camb`` is refused, because a
#: gradient through a Fortran solver would come back finite, smooth and missing
#: the whole of dP/dtheta.
PK_ENV_VAR = "GGAH_PK_BACKEND"


def resolve_backend(backend=None) -> Backend:
    """Argument, then ``$GGAH_BACKEND``, then :data:`ACCURATE`.

    The default is the accurate one because a wrong fast answer is harder to
    notice than a slow one.

    ``$GGAH_PK_BACKEND`` then overrides the resolved flavour's ``pk`` field, and
    the result is re-validated -- so asking a traced flavour for a Boltzmann
    solver raises here rather than producing an object whose gradients are
    silently incomplete.
    """
    if isinstance(backend, Backend):
        b = backend
    else:
        if backend is None:
            backend = os.environ.get(ENV_VAR)
        if backend is None:
            b = ACCURATE
        else:
            key = str(backend).lower()
            if key not in BACKENDS:
                raise ValueError(
                    f"unknown backend {backend!r}; expected one of "
                    f"{sorted(BACKENDS)} or a Backend instance")
            b = BACKENDS[key]

    pk = os.environ.get(PK_ENV_VAR)
    if pk:
        from .cosmology.power import PK_BACKENDS
        key = str(pk).lower()
        if key not in PK_BACKENDS:
            raise ValueError(
                f"${PK_ENV_VAR}={pk!r} is not a linear P(k) backend; expected "
                f"one of {sorted(PK_BACKENDS)}")
        if key != b.pk:
            b = _validate(b.with_(pk=key))
    return b


#: Retired names, kept so the benchmark and existing notebooks keep importing
#: while they migrate.  ``FAST`` was renamed for what it is rather than for how
#: it feels, and ``REFERENCE`` is gone entirely: it was ACCURATE-on-CLASS, and
#: CLASS is now what ACCURATE runs, so the two coincide and a separate name for
#: the same object is a name that can drift.
_RETIRED = {
    "FAST": ("DIFFERENTIABLE", lambda: DIFFERENTIABLE),
    "REFERENCE": ("ACCURATE (CLASS is now its default P(k) backend)",
                  lambda: ACCURATE),
}


def __getattr__(name):
    if name in _RETIRED:
        replacement, get = _RETIRED[name]
        warnings.warn(
            f"ggah_mod.backend.{name} is retired; use {replacement}.",
            DeprecationWarning, stacklevel=2)
        return get()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
