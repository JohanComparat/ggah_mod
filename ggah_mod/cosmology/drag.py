r"""The drag epoch: :math:`z_d` and the BAO sound horizon :math:`r_d = r_s(z_d)`.

The drag redshift is where the baryons stop feeling the photons' drag, defined
by a drag depth of one -- Hu & Sugiyama (1996), Sec. 4 and Eq. C-8:

.. math::

    \tau_d(z_d) = \int_0^{z_d} \frac{d\tau_T}{dz'}\,\frac{dz'}{R(z')} = 1,
    \qquad R = \frac{3\rho_b}{4\rho_\gamma},

with :math:`\tau_T` the Thomson depth -- the definition CLASS and CAMB both
implement.  It needs the free-electron history, so a recombination code and a
helium fraction, which this package does not carry.

**So** :math:`z_d` **is a calibrated fit and** :math:`r_s` **is not.**
:func:`z_drag` is a polynomial in :math:`(\ln\omega_b, \ln\omega_{cb},
\ln T_{\rm CMB}, \Sigma m_\nu)` fitted to CLASS 3.3.4 over the ``emu_pk``
training box (``tools/calibrate_zdrag.py``), and :func:`r_drag` puts it through
:func:`~ggah_mod.cosmology.background.sound_horizon`, the exact integral over
this package's own :math:`E(z)`.  The split is where the difficulty is: given
CLASS's :math:`z_d` the integral reproduces CLASS's ``rs_d`` to 1e-7, and
:math:`\partial\ln r_d/\partial z_d = -6\times10^{-4}`, so only :math:`z_d`
needed fitting.  The fit's accuracy is in
:data:`~ggah_mod.cosmology._zdrag_coefficients.VALIDATION`.

**The BBN assumptions come with the fit**, in
:data:`~ggah_mod.cosmology._zdrag_coefficients.BBN`: the helium fraction is the
one CLASS interpolates from ``sBBN_2017.dat`` -- PArthENoPE, neutron lifetime
880.2 s -- at :math:`\omega_b` and a fixed :math:`N_{\rm eff}`, so it is a
function of :math:`\omega_b` alone here and is not a parameter.  Freeing
:math:`Y_{\rm He}` or :math:`N_{\rm eff}` needs a refit with those axes.
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp

from .background import hubble_e, sound_horizon
from .parameters import Cosmology

__all__ = ["z_drag", "r_drag", "dark_energy_fraction_at_drag",
           "drag_domain_violations", "check_drag_domain"]

#: Where :func:`dark_energy_fraction_at_drag` is evaluated: the drag epoch of
#: the fiducial, to the precision a domain bound needs.
Z_DE_REF = 1060.0

#: The fit's inputs, in order.  ``omega_cb`` is the cold density
#: :attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_cb` :math:`h^2`; the
#: neutrinos enter through ``sum_mnu`` instead.
FEATURES = ("ln_omega_b", "ln_omega_cb", "ln_T_cmb", "sum_mnu")


def terms(degree: int) -> list[tuple[int, ...]]:
    """Every monomial of total degree <= ``degree`` in :data:`FEATURES`.

    As index tuples, constant first: ``()``, ``(0,)``, ..., ``(0, 0)``, ``(0, 1)``,
    ...  One ordering for the fit and the evaluation, so a coefficient file
    cannot be read against a different basis.
    """
    out: list[tuple[int, ...]] = []
    for d in range(degree + 1):
        out.extend(itertools.combinations_with_replacement(range(len(FEATURES)), d))
    return out


def design(x, degree: int):
    """``(..., n_terms)`` monomials of the feature array ``x`` (last axis features).

    Products of columns rather than powers: :math:`x^0` by ``**`` has a
    ``nan`` gradient at :math:`x = 0`, which is where ``sum_mnu`` sits for a
    massless cosmology.
    """
    cols = []
    for t in terms(degree):
        m = jnp.ones_like(x[..., 0])
        for i in t:
            m = m * x[..., i]
        cols.append(m)
    return jnp.stack(cols, axis=-1)


def features(cosmo: Cosmology, centre) -> jnp.ndarray:
    """The fit's input vector for ``cosmo``, about ``centre`` (in :data:`FEATURES` order)."""
    h2 = cosmo.h ** 2
    return jnp.stack([
        jnp.log(cosmo.Omega_b * h2 / centre[0]),
        jnp.log(cosmo.Omega_cb * h2 / centre[1]),
        jnp.log(cosmo.T_cmb / centre[2]),
        cosmo.sum_mnu - centre[3],
    ])


@functools.lru_cache(maxsize=None)
def _fit():
    """The generated coefficient module, imported on first use.

    Lazy so that ``tools/calibrate_zdrag.py`` can import :func:`terms` and
    :func:`design` before the file it writes exists.
    """
    from . import _zdrag_coefficients as K
    return K


def dark_energy_fraction_at_drag(cosmo: Cosmology):
    r"""Dark energy's share of the density at :data:`Z_DE_REF`,
    :math:`\Omega_{
m DE}f_{
m DE}(z)/E^2(z)`.

    The fit's inputs describe the expansion at the drag epoch through matter,
    radiation and neutrinos only.  Where :math:`1 + w_0 + w_a > 0` dark energy
    grows into the past, and in part of the ``emu_pk`` box it is most of the
    density at recombination -- 70 per cent at :math:`w_0 = -0.5`,
    :math:`w_a = 0.6` -- which moves :math:`z_d` by up to 45.  So the
    calibration box has this as an axis, :data:`~ggah_mod.cosmology._zdrag_coefficients.F_DE_MAX`,
    rather than fitting a regime nothing has measured.  It is negative when
    :math:`\Omega_{
m DE} < 0`, which the same box also contains.
    """
    one_z = 1.0 + Z_DE_REF
    f_de = one_z ** (3.0 * (1.0 + cosmo.w0 + cosmo.wa)) * jnp.exp(
        -3.0 * cosmo.wa * Z_DE_REF / one_z)
    return cosmo.Omega_de * f_de / hubble_e(Z_DE_REF, cosmo) ** 2


def _concrete(x):
    """``float(x)``, or ``None`` under tracing -- how the domain check is
    skipped inside ``jax.jit`` rather than breaking it (``emu_pk.interp.concrete``)."""
    try:
        return float(x)
    except Exception:
        return None


def drag_domain_violations(cosmo: Cosmology) -> dict:
    """``{name: (value, (lo, hi))}`` for every calibration axis ``cosmo`` is outside.

    Empty when inside, and for any value that is a tracer.  The axes are the
    ones the calibration sampled, :data:`~ggah_mod.cosmology._zdrag_coefficients.BOX`.
    """
    h = _concrete(cosmo.h)
    values = {
        "omega_b": None if h is None else _concrete(cosmo.Omega_b * cosmo.h ** 2),
        "omega_cdm": None if h is None else _concrete(cosmo.Omega_cdm * cosmo.h ** 2),
        "h": h,
        "sum_mnu": _concrete(cosmo.sum_mnu),
        "w0": _concrete(cosmo.w0),
        "wa": _concrete(cosmo.wa),
        "Omega_k": _concrete(cosmo.Omega_k),
        "T_cmb": _concrete(cosmo.T_cmb),
    }
    out = {}
    for name, (lo, hi) in _fit().BOX.items():
        v = values[name]
        if v is not None and not lo <= v <= hi:
            out[name] = (v, (lo, hi))
    f = _concrete(dark_energy_fraction_at_drag(cosmo))
    f_max = _fit().F_DE_MAX
    if f is not None and not abs(f) < f_max:
        out["dark_energy_fraction_at_drag"] = (f, (-f_max, f_max))
    return out


def check_drag_domain(cosmo: Cosmology) -> None:
    """Raise if ``cosmo`` is outside the box :func:`z_drag` was calibrated on.

    Names every offending axis.  Skipped for tracer values, as
    ``emu_pk.box.check`` is: a check that raised inside a trace would break the
    gradient it guards.
    """
    bad = drag_domain_violations(cosmo)
    if bad:
        raise ValueError(
            "outside the box the drag-redshift fit was calibrated on, where it "
            "extrapolates with no accuracy guarantee: "
            + "; ".join(f"{p} = {v:.5g} not in [{lo:g}, {hi:g}]"
                        for p, (v, (lo, hi)) in sorted(bad.items()))
            + ".")


@jax.jit
def _z_drag(cosmo: Cosmology):
    K = _fit()
    x = features(cosmo, jnp.asarray(K.CENTRE))
    return design(x, K.DEGREE) @ jnp.asarray(K.COEFFS)


def z_drag(cosmo: Cosmology):
    r"""The drag redshift :math:`z_d`, from the fit to CLASS.

    Accurate to 0.017 held-out and 0.029 at the box's corners
    (:data:`~ggah_mod.cosmology._zdrag_coefficients.VALIDATION`) -- 1e-5 and
    2e-5 in :math:`r_d` -- inside the calibration box and where dark energy is
    below :data:`~ggah_mod.cosmology._zdrag_coefficients.F_DE_MAX` of the
    density at the drag epoch; :func:`check_drag_domain` enforces both on
    concrete values.  Differentiable in every leaf.  It does not depend on
    ``n_s`` or ``ln10A_s``, and not on ``w0``, ``wa`` or ``Omega_k`` either:
    inside the domain their effect on :math:`z_d` is within that error, which
    the calibration measured; outside it dark energy moves :math:`z_d` by up
    to 2, which is why the domain stops there.
    """
    check_drag_domain(cosmo)
    return _z_drag(cosmo)


def r_drag(cosmo: Cosmology):
    r"""The BAO sound horizon :math:`r_d = r_s(z_d)` **[Mpc/h]**.

    :func:`~ggah_mod.cosmology.background.sound_horizon` at :func:`z_drag`.  In
    Mpc/h like every distance in the package, so :math:`D_M/r_d` needs no
    :math:`h`; divide by ``cosmo.h`` for the Mpc that CLASS's ``rs_d``, CAMB's
    ``rdrag`` and the BAO literature quote.
    """
    return sound_horizon(z_drag(cosmo), cosmo)[0]
