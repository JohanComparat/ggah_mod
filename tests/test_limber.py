r"""Verification: the angular projection, its kernels, and its beams.

The headline is closed-form.  With a :math:`\delta`-function :math:`n(z)` at
:math:`\chi_0` and a power-law spectrum,

.. math::  C_\ell = \frac{1}{\chi_0^2}\,P\!\Big(\frac{\ell+\tfrac12}{\chi_0}\Big)
                    \int W^2 d\chi

exactly -- and the test gets it to 4e-5, the residual being the finite width of
the "delta function".  That is where half-integer and :math:`1/\chi^2` errors
live, and it pins the extended-Limber convention rather than leaving it as a
comment.

The second is the beam rule.  ``B_ell`` multiplies **once per leg**, so an
auto-spectrum carries it squared.  The predecessor states that and then breaks
it: its ``C_l^{yy}`` squares the beam and its ``C_l^{XX}`` -- also an auto, in
the same file -- applies it once.  Here the rule is structural: ``c_ell`` takes
one beam per kernel and cannot do it any other way.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.backend import ACCURATE, DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, background
from ggah_mod.observables import beams, kernels
from ggah_mod.observables.limber import c_ell, limber_k

ELL = jnp.asarray([10.0, 100.0, 1000.0])
#: A narrow n(z): narrow enough that the closed form applies, wide enough that
#: the chi grid resolves it.
Z0, SIGMA_Z = 0.5, 0.01
#: The power-law index of the test spectrum.
N_INDEX = -1.5


@pytest.fixture(scope="module")
def grid():
    return kernels.limber_grid(PLANCK18, z_max=2.0, n=512)


@pytest.fixture(scope="module")
def spectrum(grid):
    z, _ = grid
    k = jnp.logspace(-4.0, 2.0, 512)
    return k, jnp.broadcast_to(k[None, :] ** N_INDEX, (z.size, k.size))


@pytest.fixture(scope="module")
def counts(grid):
    z, chi = grid
    nz = jnp.exp(-0.5 * ((z - Z0) / SIGMA_Z) ** 2)
    return kernels.number_counts(z, chi, nz)


class TestTheClosedForm:
    def test_a_delta_function_source_reproduces_it(self, grid, spectrum, counts):
        z, chi = grid
        k, stack = spectrum
        got = c_ell(ELL, k, stack, counts, counts)

        chi0 = float(jnp.atleast_1d(
            background.comoving_distance(Z0, PLANCK18))[0])
        norm = float(jnp.trapezoid(counts.w ** 2, chi))
        want = np.asarray((np.asarray(ELL) + 0.5) / chi0) ** N_INDEX \
            / chi0 ** 2 * norm
        assert np.allclose(np.asarray(got), want, rtol=1e-3)

    def test_it_is_the_extended_relation(self, grid):
        r""":math:`k = (\ell+\tfrac12)/\chi`, not :math:`\ell/\chi`.  A genuine
        :math:`O(\ell^{-2})` correction that costs nothing, and which the
        predecessor uses in one function and drops in the next."""
        _, chi = grid
        k = limber_k(jnp.asarray([10.0]), chi)
        assert float(k[0, 0]) == pytest.approx(10.5 / float(chi[0]), rel=1e-12)

    def test_the_kernels_must_share_a_grid(self, grid, spectrum, counts):
        """A pair projected on different grids has an inconsistent
        cross-covariance and nothing downstream can tell."""
        z, chi = grid
        other = kernels.number_counts(z[::2], chi[::2],
                                      jnp.ones_like(z[::2]))
        k, stack = spectrum
        with pytest.raises(ValueError, match="different\n?.*chi grids|chi grids"):
            c_ell(ELL, k, stack, counts, other)


class TestTheKernels:
    def test_number_counts_is_normalised_over_chi(self, grid, counts):
        r"""Over :math:`\chi`, because that is the variable the integral runs
        in.  Normalising over :math:`z` and integrating over :math:`\chi` is a
        factor of :math:`dz/d\chi` that no shape comparison reveals."""
        _, chi = grid
        assert float(jnp.trapezoid(counts.w, chi)) == pytest.approx(1.0,
                                                                    rel=1e-10)

    def test_the_lensing_efficiency_vanishes_behind_the_sources(self, grid):
        z, chi = grid
        nz = jnp.exp(-0.5 * ((z - 1.0) / 0.05) ** 2)
        w = np.asarray(kernels.lensing_efficiency(z, chi, nz, PLANCK18).w)
        behind = np.asarray(z) > 1.3
        assert np.all(w[behind] < 1e-3 * w.max())
        assert float(np.asarray(z)[np.argmax(w)]) < 1.0     # peaks in front

    def test_the_geometry_is_a_where_and_not_a_clip(self):
        r"""``jnp.clip((chi_s-chi_l)/chi_s, 0, None)`` -- which both of the
        predecessor's two lensing kernels use -- splits a gradient 50/50 at its
        tie, and the tie is :math:`\chi_s = \chi_l`, which every source
        distribution passes through by construction."""
        import inspect
        body = inspect.getsource(kernels.lensing_efficiency).split('"""')[-1]
        # Comments stripped: the body *explains* why it is not a clip, and the
        # explanation is the thing worth keeping.
        code = "\n".join(l.split("#")[0] for l in body.splitlines())
        assert "jnp.where" in code
        assert "clip" not in code

    def test_there_is_one_lensing_efficiency(self):
        """The predecessor has two, in two files, quadratured differently."""
        import inspect
        src = inspect.getsource(kernels)
        assert src.count("def lensing_efficiency") == 1

    def test_cmb_lensing_uses_a_single_source_plane(self, grid):
        z, chi = grid
        w = np.asarray(kernels.cmb_lensing(z, chi, PLANCK18).w)
        assert np.all(w > 0.0)                # every lens is in front of z*
        assert kernels.Z_STAR > 1000.0

    def test_the_tsz_window_is_unity_and_says_why(self, grid):
        z, chi = grid
        y = kernels.thermal_sz(z, chi)
        assert np.all(np.asarray(y.w) == 1.0)
        assert "already in" in kernels.thermal_sz.__doc__

    def test_a_kernel_is_a_pytree_with_no_string_leaf(self, counts):
        leaves = jax.tree_util.tree_leaves(counts)
        assert len(leaves) == 3
        assert all(not isinstance(x, str) for x in leaves)


class TestTheBeamRule:
    def test_one_factor_per_leg(self, grid, spectrum, counts):
        """An auto-spectrum built from one field passed twice carries
        ``B_ell^2``.  Structural: `c_ell` takes one beam per kernel."""
        k, stack = spectrum
        b = beams.gaussian(ELL, 30.0)
        bare = c_ell(ELL, k, stack, counts, counts)
        one = c_ell(ELL, k, stack, counts, counts, beam_a=b)
        two = c_ell(ELL, k, stack, counts, counts, beam_a=b, beam_b=b)
        assert np.allclose(np.asarray(one), np.asarray(bare) * np.asarray(b),
                           rtol=1e-12)
        assert np.allclose(np.asarray(two),
                           np.asarray(bare) * np.asarray(b) ** 2, rtol=1e-12)

    def test_a_beam_only_suppresses(self, grid, spectrum, counts):
        k, stack = spectrum
        bare = np.asarray(c_ell(ELL, k, stack, counts, counts))
        beamed = np.asarray(c_ell(ELL, k, stack, counts, counts,
                                  beam_a=beams.gaussian(ELL, 30.0)))
        assert np.all(beamed <= bare)
        assert beamed[-1] < bare[-1]          # and it bites at high ell


class TestTheKingPsf:
    r"""The closed-form case, and the parameter it frees."""

    def test_alpha_three_halves_is_the_elementary_case(self):
        r""":math:`K_{1/2}` makes :math:`B_\ell = e^{-\ell\theta_c}` -- the only
        case the predecessor could compute, because it reached for
        ``scipy.special.kv``."""
        ell = jnp.asarray([10.0, 100.0, 1000.0, 5000.0])
        got = beams.king(ell, 8.64, 1.5)
        want = jnp.exp(-ell * 8.64 * beams.ARCSEC)
        assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-6)

    def test_it_tends_to_one_at_large_scales(self):
        assert float(beams.king(jnp.asarray([0.0]))[0]) == pytest.approx(1.0)

    @pytest.mark.x64
    @pytest.mark.parametrize("name,x0", [("theta_c", 8.64), ("alpha", 1.5)])
    def test_it_differentiates_in_both_parameters(self, name, x0):
        """Which is the point: the King slope stops being a static choice."""
        ell = jnp.asarray([200.0, 2000.0])

        def f(v):
            tc, al = (v, 1.5) if name == "theta_c" else (8.64, v)
            return jnp.sum(beams.king(ell, tc, al))

        h = 1e-6
        ad = float(jax.grad(f)(x0))
        fd = float((f(x0 + h) - f(x0 - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_its_wings_outlive_a_gaussians(self):
        r"""A King PSF falls as :math:`e^{-\ell\theta_c}` where a Gaussian
        falls as :math:`e^{-\ell^2\sigma^2/2}`, so far enough out the King is
        the larger by any margin -- which is the reason to model an X-ray PSF
        with one rather than fitting an effective Gaussian width.

        Near the core the comparison depends entirely on which widths are
        chosen and says nothing; at the eROSITA numbers the 30-arcsecond
        Gaussian is *less* suppressed at ell = 5000.
        """
        near, far = jnp.asarray([5000.0]), jnp.asarray([200000.0])
        assert float(beams.king(near, 8.64, 1.5)[0]) \
            < float(beams.gaussian(near, 30.0)[0])
        # exp(-l t) against exp(-l^2 s^2/2): 2.3e-4 against 6e-34.
        assert float(beams.king(far, 8.64, 1.5)[0]) \
            > 1e6 * float(beams.gaussian(far, 30.0)[0])


class TestDifferentiability:
    @pytest.mark.x64
    def test_the_gradient_flows_through_the_projection(self, grid, spectrum,
                                                       counts):
        k, stack = spectrum

        def f(amp):
            return jnp.sum(c_ell(ELL, k, amp * stack, counts, counts))

        h = 1e-6
        ad = float(jax.grad(f)(1.0))
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)

    @pytest.mark.x64
    def test_the_gradient_reaches_the_cosmology_through_the_kernel(self, grid):
        z, _ = grid

        def f(omega_m):
            cosmo = PLANCK18.replace(Omega_m=omega_m)
            chi = background.comoving_distance(z, cosmo)
            nz = jnp.exp(-0.5 * ((z - 1.0) / 0.1) ** 2)
            return jnp.sum(kernels.lensing_efficiency(z, chi, nz, cosmo).w)

        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_there_is_no_python_loop_over_redshift(self):
        """Every `angular_cl_*` in the predecessor builds its P(k,z) stack with
        a list comprehension over z, which is why d/dz does not exist there."""
        import inspect
        body = inspect.getsource(c_ell).split('"""')[-1]
        assert "jax.vmap" in body
        assert "for " not in body.replace("for beam in", "")

    def test_jit_is_value_identical(self, grid, spectrum, counts):
        k, stack = spectrum
        f = lambda s: c_ell(ELL, k, s, counts, counts)
        assert np.allclose(np.asarray(jax.jit(f)(stack)), np.asarray(f(stack)),
                           rtol=1e-12)
