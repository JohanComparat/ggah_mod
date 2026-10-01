"""The cosmological parameters, as one object with one amplitude.

Two decisions are baked in here rather than left to convention, because in the
predecessor package both were settled differently in different files and the
disagreement was invisible.

**One amplitude.**  :math:`\\ln 10^{10}A_s` is the amplitude.  ``sigma8`` and
``S8`` are *outputs* -- functions of the spectrum, computed by
:mod:`ggah_mod.cosmology.amplitude` -- and are not accepted as inputs anywhere.
A container that carried both would be over-determined, and which one won would
depend on which module you were in.  Passing one to :class:`Cosmology` raises.

**Omega_m contains the neutrinos.**  :math:`\\Omega_m = \\Omega_b +
\\Omega_{cdm} + \\Omega_\\nu`, so the budget closes by construction and
:attr:`Cosmology.Omega_cdm` is derived, never supplied.  The cold density
:math:`\\Omega_{cb} = \\Omega_m - \\Omega_\\nu` is what halos form from and is
exposed separately: the two are different numbers and the package never lets
one stand in for the other silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as _np

from . import constants as C

__all__ = ["Cosmology", "RETIRED_KEYS", "PLANCK18", "nu_energy_factor"]

#: Names that used to be inputs and are now outputs.  Supplying one is an
#: error, not a deprecation: the value would be silently ignored or silently
#: preferred, and both have happened.
RETIRED_KEYS = ("sigma8", "sigma_8", "S8", "s8", "A_s", "As")



#: Panel edges of the composite Gauss-Legendre rule for the relic energy
#: integral, in :math:`x = p/(k_BT)`, and the nodes in each panel.
#:
#: Composite and not Gauss-Laguerre, because the integrand
#: :math:`x^2\sqrt{x^2+y^2}/(e^x+1)` has a branch point at :math:`x = \pm iy`,
#: which comes down onto the real axis as the species becomes relativistic.  A
#: single Laguerre rule converges slowly there -- 3e-10 at 128 nodes, worst near
#: :math:`y \sim 0.03` -- while short panels near the origin keep the singularity
#: several panel-widths away.  Against a 30-digit quadrature over
#: :math:`y \in [0, 10^6]` this rule is within **4.2e-11** everywhere, with 54
#: nodes; 96 buy 1.2e-12 at twice the cost of every :math:`E(z)`, which no
#: comparison in the package can resolve (CLASS's own ``tol_ncdm_bg`` defaults
#: to 1e-5).  The tail beyond :math:`x = 40` is :math:`e^{-40} \approx 4\times10^{-18}`.
#:
#: The two outer panels carry twelve nodes rather than ten for one reason: the
#: cold asymptote is :math:`\sum_i W_i`, and with ten there it sat 2.7e-11 below
#: :data:`~.constants.NU_KAPPA` -- far inside the stated accuracy, and still a
#: mismatch between the integral's limit and the rest mass
#: :attr:`Cosmology.Omega_nu_matter` computes from the exact constant.  With
#: twelve the two agree to 1e-14.
_FD_EDGES = (0.0, 0.5, 2.0, 6.0, 16.0, 40.0)
_FD_NODES_PER_PANEL = (10, 10, 10, 12, 12)


def _fd_rule():
    """Nodes and weights, the weights carrying :math:`x^2/(e^x+1)` and the norm.

    Normalised so that :math:`\\sum_i W_i x_i = 1`: the relativistic energy
    moment is the unit, which makes :math:`F(0) = 1` and puts the pressureless
    asymptote at :math:`\\sum_i W_i`, which is :data:`~.constants.NU_KAPPA` to
    1e-14.  Built once with numpy, so the jaxpr carries them as constants.
    """
    xs, ws = [], []
    for a, b, n in zip(_FD_EDGES[:-1], _FD_EDGES[1:], _FD_NODES_PER_PANEL):
        xg, wg = _np.polynomial.legendre.leggauss(n)
        xs.append(0.5 * (b - a) * (xg + 1.0) + a)
        ws.append(0.5 * (b - a) * wg)
    x = _np.concatenate(xs)
    wf = _np.concatenate(ws) * x ** 2 / (_np.exp(x) + 1.0)
    return x, wf / _np.sum(wf * x)


_FD_X, _FD_W = (jnp.asarray(a) for a in _fd_rule())


def nu_energy_factor(y):
    r"""The energy of one relic neutrino species, relative to massless.

    .. math::

        F(y) = \frac{120}{7\pi^4}\int_0^\infty \dd x\,
        \frac{x^2\sqrt{x^2+y^2}}{e^{x}+1},
        \qquad x = \frac{p}{k_BT_{\rm ncdm}},\quad y = \frac{m}{k_BT_{\rm ncdm}},

    Komatsu et al. (2011) Eq. 25: the energy :math:`\sqrt{p^2+m^2}` integrated
    over the spectrum the neutrinos decoupled with, :math:`[e^{p/k_BT}+1]^{-1}`
    (Lesgourgues & Pastor 2006, Eq. 6, zero chemical potential).  That is a
    Fermi-Dirac occupation in *momentum*, frozen while they were relativistic,
    and not the equilibrium distribution of a massive gas, which would be one
    in energy; nor is ``F`` the special function called the Fermi-Dirac
    integral, :math:`F_j(\eta)`.  Hence "relic energy integral" throughout.

    It is integrated on the fixed composite rule of :data:`_FD_EDGES` to 4.2e-11,
    rather than fitted.  :math:`F \to 1` when relativistic and :math:`F \to
    \kappa y` when not, with :math:`\kappa` = :data:`~.constants.NU_KAPPA`.

    **An integral and not the Komatsu et al. (2011) fit**, which the package
    used until 0.9.8.  The fit, :math:`[1+(\kappa y)^p]^{1/p}` with
    :math:`\kappa = 0.3173`, :math:`p = 1.83`, is exact in both asymptotes and
    within 0.35 per cent between them (0.1 below at :math:`y = 2.5`, 0.35
    above at :math:`y = 10`) -- and between them is where the
    neutrinos are today.  At the fiducial mass it put their kinetic energy at
    7.1e-4 of their rest mass where this integral gives 4.6e-4, so a third of
    the relativistic remainder the expansion carried was the fit, and the same
    tail dominated every :math:`\partial/\partial\Sigma m_\nu` of the
    background.  The answer is now CLASS's, which integrates the same function.

    Normalised to 1 in the *relativistic* limit, so it is a per-species ratio
    to the massless density and :func:`nu_energy_factor_mean` averages it.

    Summed as :math:`1 + \sum_i W_i\,y^2/(\sqrt{x_i^2+y^2} + x_i)`, which is
    the same sum less its value at :math:`y = 0`, written without the
    cancellation.  So :math:`F(0) = 1` **exactly** -- a massless species is
    radiation to the last bit, as it was under the fit -- and a nearly massless
    one loses no digits to the subtraction.  Differentiable in ``y``
    everywhere, including at zero, where every term's derivative vanishes.

    One definition, here rather than in
    :mod:`~ggah_mod.cosmology.background`, because :attr:`Cosmology.Omega_de`
    closes against it and importing the other way would be a cycle.
    """
    y = jnp.asarray(y)[..., None]
    y2 = y * y
    return 1.0 + jnp.sum(_FD_W * y2 / (jnp.sqrt(_FD_X * _FD_X + y2) + _FD_X),
                         axis=-1)


def nu_energy_factor_mean(y):
    r"""The species mean :math:`\tfrac1N\sum_j F(y_j)`.

    **A mean and not a sum, and the amplitude is the reason.**  The massive
    states' term has amplitude :attr:`Cosmology.Omega_nu_massive_rel`, the
    relativistic density of *all three* together, and :func:`nu_energy_factor`
    is normalised to 1 in the relativistic limit -- so it is a per-species
    ratio.  Each species carries a third of the amplitude and the density is
    their mean times the whole.  A sum would be three times the universe's
    neutrinos.

    Reduces the **last** axis, so a leading redshift shape passes through
    untouched.  That matters:
    :func:`~ggah_mod.cosmology.background.comoving_distance` hands
    :func:`~ggah_mod.cosmology.background.hubble_e` a ``z`` of shape
    ``(Nz, 256)``, and a species axis reaching the return value would arrive at
    a ``jnp.sqrt`` that expects none.
    """
    return jnp.mean(nu_energy_factor(y), axis=-1)


#: Newton steps in :func:`_lightest_mass`.
#:
#: Six reach the double-precision floor; eight are carried for margin, at
#: sixteen flops.  Swept over the whole domain in both orderings the worst
#: relative residual is 3.7e-4 after three steps, 5.3e-7 after four, 1.1e-12
#: after five and 2.9e-16 after six, after which further steps are idempotent.
#: Convergence is slowest at the floor, where the root goes to zero.
_NU_NEWTON_STEPS = 8


@jax.custom_jvp
def _lightest_mass(sum_mnu, offset_a, offset_b):
    r"""The lightest eigenstate, solving :math:`x+\sqrt{x^2+A}+\sqrt{x^2+B}=\Sigma`.

    **Newton from the degenerate answer, and that starting point is a proof
    rather than a guess.**  :math:`\sqrt{x^2+A}\ge x`, so
    :math:`\Sigma = g(x_*)\ge 3x_*` and :math:`x_0 = \Sigma/3` is an upper
    bound on the root.  :math:`g` is strictly increasing and convex
    (:math:`g'' = A(x^2+A)^{-3/2}+B(x^2+B)^{-3/2} > 0`), so Newton started
    above the root of a convex increasing function decreases monotonically and
    stays above it.  It cannot overshoot into a negative mass, there is nothing
    to bracket, and no iterate can divide by zero:
    :math:`g' = 1 + x/r_A + x/r_B` with :math:`r_A, r_B \ge \sqrt A > 0`
    wherever :math:`A > 0`, and where :math:`A = B = 0` the whole expression
    collapses to :math:`x = \Sigma/3` on the first step and stays there.

    A bisection would also be correct and would have **zero gradient**, which
    is the whole reason it is not used here.
    """
    x = sum_mnu / C.N_NU_MASSIVE
    for _ in range(_NU_NEWTON_STEPS):
        r_a = jnp.sqrt(x * x + offset_a)
        r_b = jnp.sqrt(x * x + offset_b)
        g = x + r_a + r_b - sum_mnu
        # `1 + x/r` is g'.  Where A = B = 0 this is 3 and the step is exact.
        gp = 1.0 + jnp.where(r_a > 0.0, x / r_a, 1.0) \
                 + jnp.where(r_b > 0.0, x / r_b, 1.0)
        x = x - g / gp
    return x


@_lightest_mass.defjvp
def _lightest_mass_jvp(primals, tangents):
    r"""Implicit differentiation: :math:`g(x_*)=\Sigma \Rightarrow dx_*/d\Sigma = 1/g'`.

    Exact rather than merely converged, exact in the second derivative too
    (which a Fisher forecast takes), and it removes the reverse-mode tape for
    eight square-root-heavy steps.

    **It has a value at the floor**, which is what the corner needed: there
    :math:`x_*=0` and :math:`g'(0)=1`, so :math:`dx_*/d\Sigma = 1` -- the
    lightest state absorbs the whole of a small increase in the sum while the
    two heavy states do not move, :math:`dm/d\Sigma = x/\sqrt{x^2+A} = 0`.
    That is the physics, and it is finite.
    """
    s, a, b = primals
    ds, _, _ = tangents          # A and B are constants; no tangent flows
    x = _lightest_mass(s, a, b)
    r_a, r_b = jnp.sqrt(x * x + a), jnp.sqrt(x * x + b)
    gp = 1.0 + jnp.where(r_a > 0.0, x / r_a, 1.0) \
             + jnp.where(r_b > 0.0, x / r_b, 1.0)
    return x, ds / gp


def _nu_offsets(hierarchy):
    """``NU_OFFSETS[hierarchy]``, with the ``KeyError`` turned into a sentence.

    Reached inside a ``jit`` trace -- which is precisely why ``nu_hierarchy``
    is *static*: it indexes a dict by name, and a traced string is not a
    string.  A bare ``KeyError`` from four frames inside a jaxpr is not a
    refusal anyone can act on.
    """
    try:
        return C.NU_OFFSETS[hierarchy]
    except (KeyError, TypeError):
        raise ValueError(
            f"nu_hierarchy must be one of {sorted(C.NU_OFFSETS)}, got "
            f"{hierarchy!r}.") from None


def _concrete(x):
    """``float(x)`` if ``x`` is a concrete value, else ``None``.

    The same escape :func:`~ggah_mod.cosmology.power._check_curvature` uses,
    and for the same reason: :meth:`Cosmology.replace` is called with a tracer
    under ``jax.grad`` by this package's own tests, and a check that raised
    there would kill the gradient it exists to protect.
    """
    try:
        return float(x)
    except Exception:
        return None


def nu_masses(sum_mnu, nu_hierarchy="normal"):
    r"""The three eigenstate masses [eV], **ascending**, shape ``(3,)``.

    Two numbers determine three: the sum and the ordering, through the
    splittings of :data:`~ggah_mod.cosmology.constants.NU_OFFSETS`.  So the
    masses are derived and never stored -- storing all five would let them
    disagree.

    Ascending rather than labelled :math:`(m_1,m_2,m_3)`, because the reduction
    that consumes them is a mean and does not care, while every consumer that
    *does* care wants them sorted: CLASS's ``m_ncdm`` list, CAMB's
    ``nu_mass_fractions``, and ``emu_pk`` 2.0's ordered simplex.  In a normal
    ordering ascending **is** :math:`(m_1,m_2,m_3)`; in an inverted one it is
    :math:`(m_3,m_1,m_2)`, and that relabelling is the whole of the difference
    between the two spellings of the same three numbers.

    Below the ordering's floor there is no solution and :meth:`Cosmology.create`
    refuses.  What is returned there is the degenerate :math:`\Sigma m_\nu/3`:
    finite, smooth, and unreachable through a validated construction.  It
    exists so that a cosmology rebuilt positionally inside a ``jit`` trace --
    which is what ``tree_unflatten`` and ``Cosmology(*key)`` both do -- cannot
    produce a NaN that surfaces four layers away as a dead gradient.
    """
    a, b = _nu_offsets(nu_hierarchy)
    floor = C.NU_MASS_FLOOR[nu_hierarchy]
    above = sum_mnu >= floor
    # `where` and not `maximum`: `maximum`'s derivative at a tie is 1/2, and
    # the tie is the floor, which the test ladder evaluates.
    solvable = jnp.where(above, sum_mnu, floor)
    x = _lightest_mass(solvable, a, b)
    split = jnp.stack([x, jnp.sqrt(x * x + a), jnp.sqrt(x * x + b)])
    degenerate = jnp.broadcast_to(sum_mnu / C.N_NU_MASSIVE, (C.N_NU_MASSIVE,))
    return jnp.where(above, split, degenerate)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Cosmology:
    """A w0waCDM cosmology, flat or curved, with three massive neutrinos.

    A JAX pytree: every field is a differentiable leaf, so ``jax.grad`` of
    anything downstream flows through the cosmology without a dict in the way.

    Parameters
    ----------
    Omega_m : float
        Total matter density **including neutrinos**.
    Omega_b : float
        Baryon density.
    h : float
        :math:`H_0/(100\\,\\mathrm{km\\,s^{-1}\\,Mpc^{-1}})`.
    n_s : float
        Scalar spectral index.
    ln10A_s : float
        :math:`\\ln 10^{10}A_s`.  The only amplitude.
    sum_mnu : float
        :math:`\\Sigma m_\\nu` [eV].  Split over
        :data:`~ggah_mod.cosmology.constants.N_NU_MASSIVE` eigenstates by
        :attr:`nu_hierarchy`, which are at
        :data:`~ggah_mod.cosmology.constants.T_NCDM_OVER_T_GAMMA`.
    Omega_k : float
        Curvature today.  Zero is flat and is the default; positive is open.
        See :attr:`Omega_de`, which closes against it, and
        :func:`~ggah_mod.cosmology.background.transverse_distance`, which is
        where it stops being a bookkeeping entry and becomes geometry.
    w0, wa : float
        CPL dark energy equation of state, :math:`w(a) = w_0 + w_a(1-a)`.
    T_cmb : float
        CMB temperature [K].  Sets the radiation and neutrino backgrounds.
    """

    Omega_m: float = 0.31
    Omega_b: float = 0.0493
    h: float = 0.6736
    n_s: float = 0.9649
    ln10A_s: float = 3.044
    sum_mnu: float = 0.06
    w0: float = -1.0
    wa: float = 0.0
    Omega_k: float = 0.0
    T_cmb: float = C.T_CMB
    nu_hierarchy: str = "normal"          # STATIC -- see STATIC_FIELDS

    # ---------------------------------------------------------------- derived
    @property
    def Omega_nu(self):
        r"""The neutrinos' **rest mass** density today, :math:`\Omega_\nu^{\rm nr}`.

        The same number as :attr:`Omega_nu_matter`, by definition rather than by
        agreement: it *is* that property.  At :data:`~.constants.T_CMB` it is
        :math:`\Sigma m_\nu/(93.1434\,{\rm eV}\,h^2)`, with the denominator
        :data:`~.constants.NU_DENOM_EV` derived from the same expression.

        It used to be a second density, :math:`\Sigma m_\nu/(93.14\,h^2)` with
        the 93.14 typed, beside an :attr:`Omega_nu_matter` computed at
        :math:`\Sigma m_\nu/92.717`, and the paper had to explain an
        :math:`\Omega_\nu(0)/\Omega_\nu = 1.0053` that was the ratio of two
        conventions.  There is one now, and it is CLASS's.
        """
        return self.Omega_nu_matter

    @property
    def Omega_nu_rel(self):
        r"""The density the neutrinos would carry massless, :math:`\tfrac78(4/11)^{4/3}N_{\rm eff}\Omega_\gamma`.

        What they do carry at early times whatever their masses, and the sum of
        :attr:`Omega_nu_massive_rel` and :attr:`Omega_ur`.  It does not vanish
        with the mass, which is the whole point: :math:`\Sigma m_\nu = 0` is a
        relativistic species, not an absent one.

        **Not** the relativistic *part* of today's density, which is
        :attr:`Omega_nu_r`.
        """
        return C.NU_REL_COEF * C.N_EFF * self.Omega_gamma

    @property
    def Omega_nu_massive_rel(self):
        r"""The three massive states counted as if massless, :math:`\tfrac78(4/11)^{4/3}\,3(T_{\rm ncdm}/T_\nu)^4\,\Omega_\gamma`.

        The amplitude :func:`nu_energy_factor` multiplies.  Short of
        :attr:`Omega_nu_rel` by :attr:`Omega_ur`.
        """
        return C.NU_REL_COEF * C.N_MASSIVE_EFF * self.Omega_gamma

    @property
    def Omega_ur(self):
        r"""The massless remainder, :math:`\tfrac78(4/11)^{4/3}\,\Delta N_{\rm ur}\,\Omega_\gamma`.

        :math:`\Delta N_{\rm ur}` = :data:`~.constants.N_UR_REMAINDER` = 0.0044,
        the part of :math:`N_{\rm eff}` the massive states at
        :data:`~.constants.T_NCDM_OVER_T_GAMMA` do not carry.  CLASS's
        ``N_ur``.  Radiation at every redshift, and neutrino density, so it is
        part of :attr:`Omega_nu_today` and of :attr:`Omega_nu_r`.

        Written as the difference of the two relativistic densities rather than
        as its own product: the two are within a factor of two, so the
        subtraction is exact (Sterbenz), and a massless model's
        :attr:`Omega_nu_today` is then :attr:`Omega_nu_rel` to the last bit.
        """
        return self.Omega_nu_rel - self.Omega_nu_massive_rel

    @property
    def Omega_nu_today(self):
        r""":math:`\Omega_\nu = \Omega_\nu^{\rm nr} + \Omega_\nu^{\rm r}` today, as the expansion carries it.

        :attr:`Omega_nu_massive_rel` times the species mean of
        :func:`nu_energy_factor`, plus :attr:`Omega_ur`.  At
        :math:`\Sigma m_\nu = 0.06` eV it sits 5.0e-4 above
        :attr:`Omega_nu_matter` in the degenerate spelling: 4.6e-4 of it the
        three states' kinetic energy, 0.4e-4 the massless remainder.  That is
        the whole of the difference, and all of it is physics.

        Averaged over the three species, because :math:`F` is **nonlinear** in
        :math:`y` -- so unlike :attr:`Omega_nu_matter` this one moves when the
        masses are split: a normal ordering at the fiducial mass leaves the
        lightest state, at :math:`y_0 = 5.6`, still partly relativistic.
        """
        return (self.Omega_nu_massive_rel * nu_energy_factor_mean(self.nu_y)
                + self.Omega_ur)

    @property
    def Omega_nu_r(self):
        r"""The **relativistic** part of today's neutrino density, :math:`\Omega_\nu^{\rm r}`.

        :attr:`Omega_nu_today` less :attr:`Omega_nu_matter`: the massive
        states' kinetic energy plus the massless remainder.  It is what
        :attr:`Omega_de` absorbs beyond :math:`1 - \Omega_\gamma - \Omega_m -
        \Omega_k`, and what the closure of Sec. 2.1 of the technical paper
        names beside :math:`\Omega_\nu^{\rm nr}`.
        """
        return self.Omega_nu_today - self.Omega_nu_matter

    @property
    def nu_masses(self):
        r"""The three eigenstate masses [eV], ascending, shape ``(3,)``.

        :func:`nu_masses` evaluated at this cosmology's
        :attr:`sum_mnu` and :attr:`nu_hierarchy`.
        """
        return nu_masses(self.sum_mnu, self.nu_hierarchy)

    @property
    def nu_y(self):
        r""":math:`y_j = m_j/(k_B T_{\rm ncdm,0})`, one per species.  Shape ``(3,)``.

        The vector form of :attr:`nu_y0`, which is its mean.  Both exist
        because the two neutrino densities need different things:
        :attr:`Omega_nu_today` is nonlinear in :math:`y` and needs all three,
        :attr:`Omega_nu_matter` is linear and needs only the mean.
        """
        return self.nu_masses * C.NU_Y_PER_EV * (C.T_CMB / self.T_cmb)

    @property
    def nu_y0(self):
        r""":math:`y_0 = m/(k_B T_{\rm ncdm,0})`, the **species mean**, today.

        Written in closed form as :math:`(\Sigma m_\nu/N)` rather than as
        ``mean(nu_y)``, and that is deliberate.  The two are equal
        mathematically -- the mean of the split masses is the sum over three --
        but in float64 they differ by up to an ulp, and this quantity feeds
        :attr:`Omega_nu_matter` and through it :attr:`Omega_cb`,
        :attr:`Omega_cdm`, :attr:`f_nu` and :attr:`rho_cold`.  Computing it
        from the sum makes the entire matter budget **bit for bit** invariant
        under the split, rather than nearly so: the halo layer cannot move, and
        the cobaya round trip, which closes on :attr:`Omega_nu_matter`, still
        closes exactly.
        """
        return ((self.sum_mnu / C.N_NU_MASSIVE) * C.NU_Y_PER_EV
                * (C.T_CMB / self.T_cmb))

    @property
    def Omega_nu_matter(self):
        r"""The neutrinos' **rest mass** today, :math:`\Omega_\nu^{\rm nr} = \Omega_\nu^{\rm massive,rel}\,\kappa\,y_0`.

        The pressureless asymptote of :func:`nu_energy_factor`: the part of the
        neutrino energy that redshifts as :math:`(1+z)^3` at every epoch, since
        it is :math:`\sum_j m_j n_j` and the number density goes as
        :math:`(1+z)^3` whatever the momenta do.  At the fiducial temperature it
        is :math:`\Sigma m_\nu/(93.1434\,{\rm eV}\,h^2)` exactly, which is how
        :data:`~.constants.NU_DENOM_EV` is defined, and it is what CLASS
        integrates for the same three masses.

        Three properties make it the right thing to subtract from
        :attr:`Omega_m`: it is **identically zero** when :attr:`sum_mnu` is; it
        is linear in the masses, so the whole matter budget is bit for bit
        invariant under the ordering; and the Boltzmann codes, handed
        :attr:`Omega_cdm`, add back this and not something else.  Smooth in
        :attr:`sum_mnu` through zero, so it differentiates.
        """
        return self.Omega_nu_massive_rel * C.NU_KAPPA * self.nu_y0

    @property
    def Omega_cb(self):
        r"""Cold density :math:`\Omega_{cb} = \Omega_m - \Omega_\nu^{\rm nr}`.

        What halos form from.  Distinct from :attr:`Omega_m`, which is what
        lenses.  Never use one where the other is meant.

        The subtracted term is :attr:`Omega_nu_matter`, the neutrinos' rest
        mass, and not the whole of :attr:`Omega_nu_today`: the relativistic
        remainder is not matter, and subtracting it would take radiation out of
        the matter budget -- :math:`3.8\times10^{-5}` of it at
        :math:`\Sigma m_\nu = 0`.

        Until 0.9.8 the rest mass here was the Komatsu fit's asymptote at
        :math:`N_{\rm eff}/3` degeneracy, :math:`\Sigma m_\nu/92.717`, while the
        Boltzmann codes added back :math:`\Sigma m_\nu/93.143` (CLASS) and
        :math:`/93.043` (CAMB): they integrated :math:`\Omega_m - 6.5\times10^{-6}`
        of total matter at the fiducial mass and :math:`- 6.5\times10^{-5}` at
        0.6 eV.  One convention, CLASS's, removes the difference for CLASS; CAMB
        is handed the same densities by :func:`~.power.camb_input`.
        """
        return self.Omega_m - self.Omega_nu_matter

    @property
    def Omega_cdm(self):
        r""":math:`\Omega_{cdm} = \Omega_m - \Omega_b - \Omega_\nu^{\rm nr}`. Derived.

        The same subtraction as :attr:`Omega_cb`, and for a sharper reason:
        this is the number handed out as ``omega_cdm`` to CLASS and ``omch2``
        to CAMB (:mod:`~ggah_mod.cosmology.power`), and to ``emu_pk``.  Each adds
        back the neutrino rest mass it computes itself for the same three masses
        at the same temperature, so the total matter each integrates is
        :attr:`Omega_m`.
        """
        return self.Omega_m - self.Omega_b - self.Omega_nu_matter

    @property
    def Omega_gamma(self):
        r"""Photon density :math:`\Omega_\gamma` today."""
        return C.OMEGA_GAMMA_H2 * (self.T_cmb / C.T_CMB) ** 4 / self.h ** 2

    @property
    def Omega_de(self):
        r"""Dark energy density from closure.

        :math:`1 - \Omega_\gamma - \Omega_{cb} - \Omega_\nu - \Omega_k`, with
        the neutrino term :attr:`Omega_nu_today` -- rest mass *and* relativistic
        part, the neutrino density the expansion actually carries.  Closing
        against what :func:`~ggah_mod.cosmology.background.hubble_e` sums is
        what keeps :math:`E(0) = 1` identically.

        Since :attr:`Omega_cb` subtracts the rest mass, this is
        :math:`1 - \Omega_\gamma - \Omega_m - \Omega_k` less
        :attr:`Omega_nu_r`, which it absorbs.  Closing against the mass budget
        rather than against what :func:`~ggah_mod.cosmology.background.hubble_e`
        sums is the other way to get this wrong, and is what hides a missing
        relativistic species until somebody evaluates :math:`E(z)` at
        :math:`\Sigma m_\nu = 0`.

        **It closes against curvature too, since PLAN.md item E2.**  Before
        that this was flatness rather than closure: :math:`\Omega_k` was not a
        parameter, and the package declined to *appear* to support one it would
        silently ignore.  The four pieces that had to move together are named in
        :func:`~ggah_mod.cosmology.background.transverse_distance`; this is the
        first, and on its own it does nothing at all -- which is why they are
        one change.
        """
        return (1.0 - self.Omega_gamma - self.Omega_cb
                - self.Omega_nu_today - self.Omega_k)

    @property
    def rho_matter(self):
        r"""Comoving :math:`\bar\rho_m` from **total** matter [(Msun/h)/(Mpc/h)^3].

        The density that lenses.  Use for :math:`\Sigma`, :math:`\Delta\Sigma`
        and the matter tracer normalisation.
        """
        return self.Omega_m * C.RHO_CRIT0

    @property
    def rho_cold(self):
        r"""Comoving :math:`\bar\rho_{cb}` from the **cold** field.

        The density that sets a halo's Lagrangian radius.  Use for
        :math:`\sigma(M)`, :math:`dn/dM` and :math:`b(M)`.
        """
        return self.Omega_cb * C.RHO_CRIT0

    @property
    def f_nu(self):
        r"""Neutrino mass fraction :math:`\Omega_\nu^{\rm nr}/\Omega_m`.

        Built on :attr:`Omega_nu_matter` so that it is the fraction of
        :attr:`Omega_m` that :attr:`Omega_cb` actually excludes, and
        :math:`\bar\rho_m/\bar\rho_{cb} - 1 = f_\nu/(1-f_\nu)` holds
        identically.
        """
        return self.Omega_nu_matter / self.Omega_m

    @property
    def hubble_distance_mpc(self):
        r""":math:`c/H_0` [Mpc] -- note **not** Mpc/h."""
        return C.C_KM_S / (100.0 * self.h)

    @property
    def hubble_distance(self):
        r""":math:`c/H_0` [Mpc/h], the package's distance unit."""
        return C.C_KM_S / 100.0

    # ----------------------------------------------------------------- pytree
    def tree_flatten(self):
        leaves = tuple(getattr(self, f) for f in LEAF_FIELDS)
        aux = tuple(getattr(self, f) for f in STATIC_FIELDS)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        # object.__setattr__ path via dataclass __init__ keeps validation out of
        # the trace: unflattening happens inside jit and must not branch on
        # values.
        #
        # Positional in the leaves, keyword in the statics.  The statics are
        # declared last, so `cls(*leaves)` would also reach the right slots --
        # the keywords are written out so that reordering the dataclass breaks
        # loudly here rather than silently assigning a temperature to a
        # hierarchy.
        return cls(*leaves, **dict(zip(STATIC_FIELDS, aux)))

    # ------------------------------------------------------------ convenience
    def replace(self, **kw) -> "Cosmology":
        """A copy with some fields changed, validated."""
        _reject_retired(kw)
        out = replace(self, **kw)
        # Validated on the *result*, not on `kw`.  `replace(sum_mnu=0.03)` on
        # an inverted cosmology is illegal and neither argument says so.
        _validate_nu(out.sum_mnu, out.nu_hierarchy)
        return out

    @classmethod
    def create(cls, **kw) -> "Cosmology":
        """Build a :class:`Cosmology`, refusing retired amplitude keys.

        Prefer this over the constructor when the keyword arguments come from
        user input, a config file or another package: the constructor is kept
        branch-free so it can run inside a ``jit`` trace.
        """
        _reject_retired(kw)
        unknown = set(kw) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise TypeError(
                f"Cosmology.create: unknown parameter(s) {sorted(unknown)}; "
                f"known are {sorted(f.name for f in cls.__dataclass_fields__.values())}")
        out = cls(**kw)
        _validate_nu(out.sum_mnu, out.nu_hierarchy)
        return out


def _reject_retired(kw: dict) -> None:
    stray = [k for k in RETIRED_KEYS if k in kw]
    if stray:
        raise TypeError(
            f"{', '.join(repr(s) for s in stray)} is not an input to this "
            f"package.  The amplitude is 'ln10A_s' and sigma8/S8 are derived "
            f"from the spectrum -- see ggah_mod.cosmology.amplitude.sigma8, "
            f".s8.  A cosmology carrying both an amplitude and a sigma8 is "
            f"over-determined, and which one wins used to depend on which "
            f"module you were in.  To start from a sigma8, solve for the "
            f"amplitude once, up front, with amplitude.ln10A_s_for_sigma8().")


#: Fields carried in the pytree's ``aux_data`` rather than as differentiable
#: leaves.  Declared **last** in the dataclass, so positional reconstruction of
#: the leaves -- ``cls(*leaves)`` here, ``Cosmology(*key)`` in
#: :mod:`~ggah_mod.cosmology.power` -- is unaffected.
#:
#: ``nu_hierarchy`` is here for two reasons that point the same way.  It indexes
#: a dictionary by name inside a ``jit`` trace
#: (:data:`~ggah_mod.cosmology.constants.NU_OFFSETS`, in :func:`nu_masses`), and
#: a tracer is not a string.  And large-scale structure responds to
#: :math:`\Sigma m_\nu` rather than to how the sum is divided, so the ordering
#: carries no likelihood gradient: it is two discrete hypotheses, chosen once at
#: the start of a run, not two points on an axis a chain explores.  The rule
#: this repository states -- *a field that is not a leaf is a parameter no
#: forecast can vary* -- is satisfied by it, not broken.
STATIC_FIELDS = ("nu_hierarchy",)

#: The differentiable leaves, in flatten order.
#:
#: **Derived, not listed.**  Every field not declared static is a leaf, so
#: adding a parameter to the dataclass makes it a leaf, puts it in the Boltzmann
#: cache key and puts it in the tests that sweep every parameter, with nothing
#: to remember.  That is the same rule :func:`~ggah_mod.cosmology.power._as_key`
#: needs, and the reason it could never take a hand-picked subset.
LEAF_FIELDS = tuple(f for f in Cosmology.__dataclass_fields__
                    if f not in STATIC_FIELDS)

assert LEAF_FIELDS + STATIC_FIELDS == tuple(Cosmology.__dataclass_fields__), (
    "static fields must be declared last: `Cosmology(*leaves, **statics)` and "
    "`Cosmology(*key)` both reconstruct positionally")


def _validate_nu(sum_mnu, nu_hierarchy) -> None:
    """The neutrino refusals, in one place and outside the trace.

    ``nu_hierarchy`` is checked unconditionally -- it is static, so it is a
    concrete string on every path including a traced one.  ``sum_mnu`` is
    checked only when it is concrete, by the same rule
    :func:`~ggah_mod.cosmology.power._check_curvature` follows: this package's
    own tests call :meth:`Cosmology.replace` with a tracer under ``jax.grad``,
    and a check that raised there would kill the gradient it exists to protect.
    What is skipped is the second call and not the first -- a value reaches a
    trace by having been put in a :class:`Cosmology` outside one.
    """
    _nu_offsets(nu_hierarchy)
    s = _concrete(sum_mnu)
    if s is None:
        return
    if s < 0.0:
        raise ValueError(f"sum_mnu = {s:g} eV is negative.")
    if nu_hierarchy == "massless":
        if s != 0.0:
            raise ValueError(
                f"nu_hierarchy='massless' carries no mass, so sum_mnu must be "
                f"0, not {s:g}.  For three equal masses summing to {s:g} eV "
                f"use nu_hierarchy='degenerate'; for the physical split use "
                f"'normal' or 'inverted'.")
        return
    floor = C.NU_MASS_FLOOR[nu_hierarchy]
    if 0.0 < s < floor:
        a, b = C.NU_OFFSETS[nu_hierarchy]
        other = "inverted" if nu_hierarchy == "normal" else "normal"
        raise ValueError(
            f"sum_mnu = {s:g} eV has no {nu_hierarchy}-ordering solution.  The "
            f"three eigenstates are m, sqrt(m^2 + {a:g}) and sqrt(m^2 + {b:g}) "
            f"[eV^2], so their sum is smallest at m = 0 and cannot fall below "
            f"{floor:.6f} eV.  The {other} ordering's floor is "
            f"{C.NU_MASS_FLOOR[other]:.6f} eV.  Each hierarchy's domain is one "
            f"interval and this value is outside it: use 'degenerate' for three "
            f"equal masses at any sum, or 'massless' for sum_mnu = 0.")


#: Planck 2018 TT,TE,EE+lowE+lensing marginalised means, at the minimal mass,
#: except :math:`\Omega_m = 0.31` where that column gives 0.3153.
#:
#: Built through :meth:`Cosmology.create` rather than the constructor: the
#: shipped default goes through the door every other caller is told to use.
#:
#: **It declares** ``nu_hierarchy="degenerate"`` **while the class default is**
#: ``"normal"``.  Neither reproduces the Planck 2018 baseline exactly: Planck
#: assumes the normal ordering at its minimal mass (their Sect. 7.5.1) and
#: approximates it as two massless states and a single massive one of 0.06 eV
#: (Sect. 2.1).  ``degenerate`` is the convention Planck adopts when it varies
#: :math:`\Sigma m_\nu`, neglecting the splittings at its sensitivity
#: (Sect. 7.5.1).
#:
#: It had a second consequence, and **that one has now expired, as it said it
#: would**.  ``emu_pk`` 1.x refused a split, so until 2026-09-10 this was also
#: the reason the differentiable flavour could run at the fiducial at all.
#: ``emu_pk`` 2.0.0 carries ``nu_r1``/``nu_r2`` and answers for any ordering, so
#: the choice is now free of that constraint -- and it is still ``degenerate``,
#: Planck's convention for a varied sum.
#:
#: A *new* cosmology gets the physics: ``Cosmology.create(sum_mnu=0.06)`` is a
#: normal ordering, and its three masses are (0.00095, 0.00871, 0.05035) eV.
PLANCK18 = Cosmology.create(nu_hierarchy="degenerate")
