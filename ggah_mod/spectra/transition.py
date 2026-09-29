r"""The one-halo/two-halo transition, as a registry.

.. math::  P_{ab}(k) = T(k)\,P^{1h}_{ab}(k) + P^{2h}_{ab}(k)

**The default is** :math:`T = 1`, the literal sum -- what pyhalomodel and Mead &
Verde (2021) use, and what CLASS's HMcode-2020 applies in its baryonic and
unfitted variants.  The transition region is repaired where it is broken, in the
two-halo term: :mod:`~ggah_mod.spectra.bnl` adds the measured beyond-linear bias,
which is 10-50 per cent of :math:`P^{2h}` across the transition.  A damping
entry stays selectable, for the one place it is justified.

What the one-halo plateau is, and where it matters
---------------------------------------------------

Without :math:`T`, :math:`P^{1h}(k\to0)` tends to a constant,
:math:`\int dM\,(dn/dM)\,W_a(0|M)W_b(0|M)`.  Three facts decide what to do about
it:

* **In configuration space it is harmless.**  :math:`\xi^{1h}(r)` has compact
  support, :math:`r < 2R_\Delta` of the largest halo, and the plateau is its
  integral.  :math:`\xi`, :math:`w_p` and :math:`\Delta\Sigma` at
  :math:`r > 0` do not see a constant in :math:`P(k)`.  A damping, by contrast,
  makes :math:`\xi^{1h}` negative from about 4 to 44 Mpc/h and moves :math:`w_p`
  by up to 13 per cent at :math:`r_p = 3` Mpc/h across its coefficient's box.
  The :math:`w_p` collapse ``ggah_cal`` once reported against the plateau was
  the Ogata reach of ``n_hankel``, and did not reproduce once that was fixed.
* **For rare haloes it is physical.**  The Poisson power of clusters is the
  large-scale one-halo term of the thermal SZ and X-ray spectra; damping it
  removes signal.
* **For the matter field it is not.**  Mass and momentum conservation make the
  total matter spectrum's halo contribution go as :math:`k^4` on large scales,
  so a Fourier-space matter spectrum at :math:`k \lesssim 0.05\,h/{\rm Mpc}`
  wants a compensation.  That is what :func:`mead20` is for.

The damping that was the default, and why it is gone
----------------------------------------------------

``"mead15"`` was :math:`x/(1+x)`, :math:`x = (k/k_*)^4`, with
:math:`k_* = 0.584/\sigma_v`.  **Neither HMcode ever used that pair.**
HMcode-2015 (and 2016) damps with :math:`1 - e^{-(k/k_*)^2}` and
:math:`k_* = 0.584/\sigma_v`; HMcode-2020 with :math:`x/(1+x)` and
:math:`k_* = 0.05618\,\sigma_{8,cb}(z)^{-1.013}\,h/{\rm Mpc}` -- the Fortran
(``halofit.f90``) and CLASS's C port (``hmcode.c``) agree.  The name now raises
and points at ``"mead20"``, the 2020 pair taken whole.  Its coefficient was
fitted jointly with HMcode-2020's two-halo damping and smoothing, which this
package does not have; that is recorded rather than refitted.

The option that was planned and is not here
-------------------------------------------

``PLAN.md`` named an entry that subtracts the :math:`k\to0` plateau.
**It is wrong, and the arithmetic says so in one line.**
:math:`|u(k|M)| \le u(0|M)` for every halo and every k, so
:math:`P^{1h}(k) \le P^{1h}(0)` *everywhere* -- the plateau is the term's
maximum, not a floor under it.  Subtracting it gives a one-halo term that is
negative at every wavenumber.  Recorded here because the idea is an easy one to
have twice.

What is fitted, and what is derived
-----------------------------------

:func:`mead20` derives its damping scale from the cosmology through
:math:`\sigma_{8,cb}(z)` of the same linear spectrum the two-halo term uses,
so only the coefficient is fitted.  That coefficient is
:class:`TransitionParams`, a declared parameter with a bound, a prior and a
reason; the exponent is a published constant and not a parameter.  It is not
in :class:`~ggah_mod.spectra.spec.PkOptions`, which is treedef material and
carries no arrays: the prescription is static and the coefficient is traced.
Passing a coefficient with ``transition="none"`` is refused, because a
parameter that is accepted and unread is a parameter that does nothing.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..cosmology.amplitude import sigma8
from ..sectors.calibration import Calibration
from ..sectors.params import Flat, Param, SectorParams, sector_params

__all__ = ["TransitionParams", "no_transition", "mead20",
           "one_halo_transition", "ONE_HALO_TRANSITION", "RETIRED",
           "TRANSITION_CALIBRATION", "K_STAR_COEFF", "K_STAR_EXPONENT"]


#: Mead et al. (2021), MNRAS 502, 1401 [arXiv:2009.01858], Eq. (17) and
#: Table 2: :math:`k_* = 0.05618\,\sigma_{8,cb}(z)^{-1.013}\,h/{\rm Mpc}`,
#: fitted against the Mira-Titan and FrankenEmu emulators.  The digits are
#: CLASS's and CAMB's (0.0561778, -1.0131066), so a comparison with either is
#: a comparison of the damping and not of rounding.
K_STAR_COEFF = 0.0561778
#: The exponent of :math:`\sigma_{8,cb}` in the same fit.  A published constant
#: of the form rather than a parameter: freeing it would make the form a fit of
#: this package's, which it is not.
K_STAR_EXPONENT = -1.0131066


@sector_params
class TransitionParams(SectorParams):
    r"""Layer 4's parameter, and until now layer 4 had none.

    One number: the coefficient in :math:`k_* = c\,\sigma_{8,cb}^{-1.013}` of
    :func:`mead20`.  It is read only when that damping is selected, which is not
    the default; with the default the parameter is refused rather than
    ignored.  It was declared because a real-data statistic isolates it:
    :math:`w_p` at :math:`r_p \approx 1\text{--}3` Mpc/h, where the one-halo
    and two-halo terms cross, moves by 13 per cent across the box and in a
    direction no HOD parameter spans.  That is also the argument for leaving the
    damping off in a :math:`w_p` fit: a damping scale that :math:`w_p`
    constrains is a damping scale :math:`w_p` is paying for.

    **The split is the same one** :class:`~ggah_mod.sectors.params.SectorParams`
    draws everywhere else, and it lines up with layer 4's existing one:
    :class:`~ggah_mod.spectra.spec.PkOptions` says *which* prescription, which is
    treedef material and triggers a recompile; this says *what value*, which is a
    traced leaf and carries a gradient.  A prescription cannot be differentiated
    into and a coefficient cannot be branched on, and neither could ever have
    lived in the other object.

    It reuses the layer-3 base class rather than growing a parallel one.  The
    name says "sector" and this is not a sector, which is a wart -- and a second
    mechanism that did the same thing would be worse, because the audits in
    ``tests/test_sectors_paths.py`` walk *containers*, not sectors, and a
    container outside their reach is precisely how ``GalaxyParams`` escaped
    ``test_every_parameter_has_a_reason_for_its_bound`` for as long as it did.
    Those audits now walk layer 4 too.
    """

    k_star_coeff: float = K_STAR_COEFF

    _PARAMS = {
        "k_star_coeff": Param(
            K_STAR_COEFF, (0.014, 0.28), Flat(), "h/Mpc",
            "sets the damping scale through k_* = c sigma8_cb(z)^-1.013; "
            "0.0561778 is Mead et al. (2021) Table 2, fitted against the "
            "Mira-Titan and FrankenEmu emulators, and gives k_* = 0.070 h/Mpc "
            "at PLANCK18 z = 0. A reasoned Flat rather than a Gaussian, because "
            "the published value was fitted jointly with HMcode-2020's two-halo "
            "damping and smoothing, which this package does not have, so "
            "quoting its uncertainty would attribute a precision to a different "
            "model. The box is the k_* window the damping can mean, 0.017 to "
            "0.35 h/Mpc, at sigma8_cb^1.013 = 0.81: below it the damping eats "
            "into the linear regime, where the two-halo term is exact to 0.2 "
            "per cent; above it the damping suppresses the one-halo term where "
            "the one-halo term is the answer",
            "physical"),
    }


def no_transition(pk, k, params: "TransitionParams" = None):
    r""":math:`T(k) = 1`: the literal sum, and the default.

    ``params`` is in the signature so every entry has one; the dispatcher
    refuses a non-``None`` value for this entry before it gets here.
    """
    return jnp.ones_like(jnp.asarray(k))


def mead20(pk, k, params: "TransitionParams" = None):
    r""":math:`T(k) = x/(1+x)`, :math:`x = (k/k_*)^4`, :math:`k_* = c\,\sigma_{8,cb}(z)^{-1.013}`.

    Mead et al. (2021), Eq. (17): HMcode-2020's one-halo damping, the form and
    the scale taken together.  The fourth power is the large-scale limit of a
    one-halo term compensated for mass and momentum conservation, which is a
    statement about the *matter* field; for a tracer of rare haloes the plateau
    this removes is physical Poisson power, so use it for Fourier-space matter
    spectra and not by default.

    :math:`\sigma_8` is taken of ``pk`` -- the cold spectrum at the field's
    redshift under the default ``two_halo_spectrum="cb"``, which is what CAMB and
    CLASS read.  Differentiable in ``params.k_star_coeff`` and in the cosmology
    through :math:`\sigma_8`, with no branch and no special function.
    """
    k = jnp.asarray(k)
    params = TransitionParams() if params is None else params
    k_star = params.k_star_coeff * sigma8(pk, k) ** K_STAR_EXPONENT
    x4 = (k / k_star) ** 4
    return x4 / (1.0 + x4)


#: The prescriptions, by name.  Each takes ``(pk, k)`` -- the same linear
#: spectrum the two-halo term is multiplying, so one decision serves both -- and
#: returns a multiplicative :math:`T(k)` of shape ``(Nk,)``.
ONE_HALO_TRANSITION = {
    "none": no_transition,
    "mead20": mead20,
}

#: Names that were prescriptions and are not, with the reason.  Checked before
#: the registry, so a configuration written for an older release fails with
#: the reason rather than with "unknown".
RETIRED = {
    "mead15": "it was HMcode-2020's (k/k*)^4/(1+(k/k*)^4) with HMcode-2015's "
              "k* = 0.584/sigma_v, a pair neither HMcode used (the 2015 form "
              "is 1 - exp(-(k/k*)^2)).  Use 'mead20', the 2020 form and scale "
              "taken together, or 'none', the default",
}

#: What each prescription was calibrated on.  Same record as the five layer-3
#: registries use, so a sixth provenance shape is not invented for a sixth
#: registry.
#:
#: ``z_range=None`` on ``"none"`` is the documented "there is no fitted range"
#: value rather than a permissive one: it is the absence of a prescription, not
#: a prescription fitted everywhere.
TRANSITION_CALIBRATION = {
    "none": Calibration(
        fit="not a fit -- the uncompensated sum",
        z_range=None,
        selection="",
        notes="The default since 0.8.0, with the beyond-linear two-halo term "
              "on. It leaves P_1h(k -> 0) at the integral of xi_1h, which has "
              "compact support: harmless in configuration space, physical "
              "Poisson power for rare-halo tracers, and a k^0 excess over the "
              "k^4 that mass conservation requires of the matter field at "
              "k < 0.05 h/Mpc.",
        cosmology_dependent=False),
    "mead20": Calibration(
        fit="Mira-Titan and FrankenEmu emulators (Mead et al. 2021, MNRAS "
            "502, 1401)",
        z_range=(0.0, 2.0),
        selection="cold matter, w0waCDM with massive neutrinos",
        notes="k_* = 0.0561778 sigma8_cb(z)^-1.013 h/Mpc, Eq. (17) and "
              "Table 2. Fitted with the rest of HMcode-2020 -- two-halo "
              "damping, transition smoothing, halo bloating -- which this "
              "package does not have, so the coefficient is that paper's for "
              "this damping and not a joint refit. The scale moves with the "
              "cosmology through sigma8; only the coefficient is fitted.",
        cosmology_dependent=True),
}


def one_halo_transition(pk, k, *, transition: str,
                        params: "TransitionParams" = None):
    r""":math:`T(k)` for the named prescription, shape ``(Nk,)``.

    The dispatcher, so the lookup and its refusals live in one place rather than
    at every call site.  An unknown name raises rather than falling back to
    ``"none"``, a retired one raises with its reason, and a coefficient passed
    to ``"none"`` raises because nothing would read it.
    """
    if transition in RETIRED:
        raise ValueError(
            f"one-halo transition {transition!r} is retired: "
            f"{RETIRED[transition]}")
    try:
        rule = ONE_HALO_TRANSITION[transition]
    except KeyError:
        raise ValueError(
            f"unknown one-halo transition {transition!r}; expected one of "
            f"{sorted(ONE_HALO_TRANSITION)}") from None
    if transition == "none" and params is not None:
        raise ValueError(
            "transition_params was passed with one_halo_transition='none', "
            "which reads no parameter; a coefficient that is accepted and "
            "unread does nothing.  Select 'mead20', or drop the argument.")
    return rule(pk, k, params)
