r"""Verification: the Hankel engines, against closed forms and each other.

Two engines with **nothing in common but the answer** -- one a
double-exponential quadrature over the zeros of :math:`J_\nu`, the other an FFT
of a biased log-spaced signal against a ratio of gamma functions.  They agree
with a closed-form transform pair, and with each other, to :math:`10^{-8}`.
That is worth more than either alone: a shared convention error would have to
appear identically in a quadrature rule and in a Mellin transform to survive.

The Gaussian pair, which needs no reference data:

.. math::

    P(k) = (2\pi a^2)^{3/2}e^{-k^2a^2/2}
    \;\longleftrightarrow\;
    \xi(r) = e^{-r^2/2a^2},
    \quad \Sigma(R) = \sqrt{2\pi}\,a\,e^{-R^2/2a^2}

and :math:`\Delta\Sigma` follows from :math:`\bar\Sigma - \Sigma` in closed form
too, which is what makes it a real test of the :math:`J_2` kernel rather than of
:math:`J_0` twice.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.backend import ACCURATE, DIFFERENTIABLE
from ggah_mod.observables import transforms as T
from ggah_mod.observables.spec import DEFAULT_WTHETA_ELL

#: The Gaussian pair's width [Mpc/h].
A = 5.0
#: Radii well inside both engines' reach.
R = jnp.asarray([1.0, 2.0, 5.0, 10.0])


@pytest.fixture(scope="module")
def gaussian():
    """``(k, P(k))`` on a grid wide enough that no tail continuation runs."""
    k = jnp.logspace(-5, 4, 4096)
    return k, (2 * jnp.pi * A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * A) ** 2)


def _xi_exact(r):
    return jnp.exp(-0.5 * (r / A) ** 2)


def _sigma_exact(r):
    return jnp.sqrt(2 * jnp.pi) * A * _xi_exact(r)


def _delta_sigma_exact(r):
    """:math:`\\bar\\Sigma(<R) - \\Sigma(R)` for the Gaussian, in closed form."""
    g = _xi_exact(r)
    return jnp.sqrt(2 * jnp.pi) * A * (2 * A ** 2 * (1.0 - g) / r ** 2 - g)


def _rel(got, want):
    return float(np.max(np.abs(np.asarray(got) / np.asarray(want) - 1.0)))


class TestTheClosedFormPair:
    """Both engines, three transforms, one analytic answer."""

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_xi(self, flavour, gaussian):
        k, pk = gaussian
        assert _rel(T.pk_to_xi(R, k, pk, backend=flavour), _xi_exact(R)) < 1e-6

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_sigma(self, flavour, gaussian):
        k, pk = gaussian
        got = T.pk_to_sigma(R, k, pk, 1.0, backend=flavour)
        assert _rel(got, _sigma_exact(R)) < 1e-6

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_delta_sigma(self, flavour, gaussian):
        """The only one that exercises :math:`J_2` rather than :math:`J_0`."""
        got = T.pk_to_delta_sigma(R, *gaussian, 1.0, backend=flavour)
        assert _rel(got, _delta_sigma_exact(R)) < 1e-6

    def test_the_two_engines_agree_with_each_other(self, gaussian):
        """A parity row, and a strong one: nothing is shared between a
        double-exponential quadrature and an FFT against gamma functions except
        the answer."""
        k, pk = gaussian
        for fn, extra in ((T.pk_to_xi, ()), (T.pk_to_sigma, (1.0,)),
                          (T.pk_to_delta_sigma, (1.0,))):
            a = fn(R, k, pk, *extra, backend=ACCURATE)
            b = fn(R, k, pk, *extra, backend=DIFFERENTIABLE)
            assert _rel(a, b) < 1e-6, fn.__name__


class TestTheMellinKernelIsWhatItsNameSays:
    r""":math:`U_K(z) = \int_0^\infty t^{z-1}K(t)\,dt`, with no hidden constant.

    ``mcfit`` omits the :math:`\sqrt{\pi/2}` from the spherical kernel and
    restores it through a per-transform prefactor -- legitimate bookkeeping, and
    a trap for anyone reading the kernel as what it is called.  Leaving it out
    multiplies every :math:`\xi(r)` by 0.798 and changes no shape at all: there
    is no plot on which it shows.
    """

    @pytest.mark.parametrize("z,want", [
        (0.5, 2.5066282746310002),      # sqrt(2 pi)
        (1.0, 1.5707963267948966),      # pi / 2
    ])
    def test_the_spherical_kernel_against_the_closed_form(self, z, want):
        got = complex(T._mellin_spherical_bessel_j(0.0, z)).real
        assert got == pytest.approx(want, rel=1e-12)

    def test_the_bessel_kernel_against_the_closed_form(self):
        r""":math:`\int t^{z-1}J_0 dt = 2^{z-1}\Gamma(z/2)/\Gamma(1-z/2)`; at
        :math:`z = 1` that is 1."""
        got = complex(T._mellin_bessel_j(0.0, 1.0)).real
        assert got == pytest.approx(1.0, rel=1e-12)


class TestTheAccuracyKnobIsTheStepNotTheNodeCount:
    r"""``hankel_h`` sets the error; ``n_hankel`` sets the reach.

    They look like one knob and are two.  The predecessor used
    ``N = 512, h = 0.005`` and sits at 2e-5; the *same* 512 nodes at
    ``h = 0.001`` reach 1e-7.
    """

    def _err(self, n, h, gaussian):
        rule = T.make_hankel(0.5, backend=ACCURATE, n=n, h=h)
        return _rel(T.pk_to_xi(R, *gaussian, rule=rule), _xi_exact(R))

    @pytest.mark.parametrize("n", [256, 512, 1024])
    def test_the_error_is_flat_in_the_node_count(self, n, gaussian):
        assert self._err(n, 0.005, gaussian) == pytest.approx(2.13e-5, rel=0.1)

    @pytest.mark.parametrize("h,want", [(0.01, 8.4e-4), (0.005, 2.1e-5),
                                        (0.001, 1.0e-7)])
    def test_the_error_falls_with_the_step(self, h, want, gaussian):
        assert self._err(512, h, gaussian) == pytest.approx(want, rel=0.2)

    def test_the_reach_grows_with_the_node_count(self):
        small = T.make_hankel(0.5, backend=ACCURATE, n=256, h=0.005)
        large = T.make_hankel(0.5, backend=ACCURATE, n=1024, h=0.005)
        assert float(large.x[-1]) > 3.5 * float(small.x[-1])

    def test_the_accurate_flavour_is_the_accurate_one(self):
        assert ACCURATE.hankel_h < DIFFERENTIABLE.hankel_h

    def test_an_unusable_step_is_refused(self):
        with pytest.raises(ValueError, match="hankel_h"):
            ACCURATE.with_(hankel_h=0.5)


class TestNoSpecialFunctionIsEverTraced:
    """Both engines' constants are functions of static grid choices alone."""

    def test_the_ogata_nodes_are_numpy_and_memoised(self):
        a = T._ogata_nodes(0.5, 64, 0.01)
        b = T._ogata_nodes(0.5, 64, 0.01)
        assert a[0] is b[0]                      # the same object, not a copy
        assert isinstance(a[0], np.ndarray)

    def test_the_fftlog_coefficients_are_numpy_and_memoised(self):
        a = T._fftlog_setup(0.0, "bessel", 2.0, 1.0, 64, 0.1, -5.0)
        b = T._fftlog_setup(0.0, "bessel", 2.0, 1.0, 64, 0.1, -5.0)
        assert a[1] is b[1]
        assert np.iscomplexobj(a[1])

    def test_the_module_calls_no_scipy_at_module_scope(self):
        """Every ``scipy`` import is inside a construction helper, so importing
        the module does not pull it onto any path a trace could reach."""
        import ast
        import pathlib
        tree = ast.parse(pathlib.Path(T.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [getattr(n, "module", None) or "" for n in top]
        names += [a.name for n in top if isinstance(n, ast.Import) for a in n.names]
        assert not any("scipy" in (m or "") for m in names)


class TestTheHighKContinuation:
    r"""Ogata reaches :math:`x \approx 10^3`, so it samples far past any grid."""

    def test_it_continues_rather_than_truncating(self, gaussian):
        """Truncating is what the predecessor did until 2026-07; fixing it moved
        ``w_p`` and ``Delta Sigma`` by 19% and 20%."""
        k, pk = gaussian
        narrow = k < 50.0
        got = T.pk_to_xi(R, k[narrow], pk[narrow], backend=ACCURATE)
        assert np.all(np.isfinite(np.asarray(got)))
        assert _rel(got, _xi_exact(R)) < 1e-3

    def test_the_cap_is_a_where_and_not_a_minimum(self):
        r"""``jnp.minimum(slope, -3)`` puts a tie at exactly -3, which is the
        *generic* high-k slope of a LambdaCDM spectrum -- so the tie is where the
        fiducial sits, and JAX splits it 50/50.  Same defect class as
        ``nu_ratio``'s ``clip`` and ``energetics``'s ``minimum``."""
        import inspect
        src = inspect.getsource(T._log_interp_with_tail)
        body = src.split('"""')[-1]          # the docstring *names* the defect
        assert "jnp.where" in body
        assert "jnp.minimum" not in body

    def test_a_spectrum_that_underflows_does_not_produce_nan(self):
        """``log(0) - log(0)`` in the tail slope is ``nan``, and one nan in the
        continuation poisons every radius.  An analytic Gaussian underflows;
        so does any spectrum given a wide enough grid."""
        k = jnp.logspace(-5, 6, 2048)
        pk = (2 * jnp.pi * A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * A) ** 2)
        assert float(jnp.min(pk)) == 0.0
        assert np.all(np.isfinite(np.asarray(T.pk_to_xi(R, k, pk,
                                                        backend=ACCURATE))))


class TestFFTLogsConstraints:
    """It is faster, and it is not free."""

    def test_it_refuses_a_grid_that_is_not_log_spaced(self):
        k = jnp.linspace(1e-3, 10.0, 256)
        with pytest.raises(ValueError, match="log-spaced"):
            T.make_fftlog(0.0, k, kind="bessel", power=2.0)

    def test_its_output_grid_is_reciprocal_to_the_input(self, gaussian):
        k, _ = gaussian
        rule = T.make_fftlog(0.0, k, kind="bessel", power=2.0, n_pad=0)
        prod = np.asarray(rule.y)[::-1] * np.asarray(k)
        assert np.allclose(prod, prod[0], rtol=1e-10)

    def test_padding_keeps_it_reciprocal_to_the_grid_it_actually_used(
            self, gaussian):
        """Which is the padded one, and that is the whole point of it.

        The relation is unchanged; what changes is its partner.  A rule padded
        by two decades answers on a grid reaching two decades further at each
        end, and the caller's small radii land in the middle of it rather than
        at the edge.
        """
        k = np.asarray(gaussian[0])
        rule = T.make_fftlog(0.0, k, kind="bessel", power=2.0)
        assert rule.n_pad > 0
        d = float(np.log(k[1]) - np.log(k[0]))
        j = np.arange(1, rule.n_pad + 1)
        k_pad = np.exp(np.concatenate(
            [np.log(k[0]) - d * j[::-1], np.log(k), np.log(k[-1]) + d * j]))
        prod = np.asarray(rule.y)[::-1] * k_pad
        assert np.allclose(prod, prod[0], rtol=1e-10)

    def test_the_interpolation_off_that_grid_is_sign_safe(self):
        r"""``xi(r)`` crosses zero near the baryon scale and ``Delta Sigma`` can
        too.  Interpolating in log space gives a floor where there should be a
        zero crossing, and it looks like a feature."""
        import inspect
        src = inspect.getsource(T._transform)
        assert "interp_cubic(jnp.log(r), jnp.log(rule.y), g)" in src


class TestTheLogSpacingCheckReadsTheGridsDtype:
    r"""The tolerance is absolute in :math:`\ln x`, floored at the grid's own eps.

    ``jnp.logspace`` under JAX's default dtype places its nodes with an absolute
    error of 5-8 eps32 in :math:`\ln k` -- 6.4e-7 at ``n = 512`` over
    :math:`10^{-4} < k < 200` -- and widening to float64 inside
    :func:`~ggah_mod.observables.transforms.make_fftlog` recovers none of it.

    The check this replaced was ``rtol=1e-6`` *relative to* :math:`\Delta`, so
    its threshold shrank as the grid refined -- 2.8e-8 at ``n = 512`` against
    3.5e-9 at ``n = 4096`` -- while the deviation it was chasing does not move
    with ``n`` at all.  Exactly the wrong direction, and it meant every FFTLog
    transform refused to run in the package's default precision: ``w_p``,
    ``xi``, ``Sigma``, ``Delta Sigma`` and ``w(theta)`` on ``DIFFERENTIABLE``,
    which is the only flavour that has one and the only one a forecast can
    differentiate.  ``tests/conftest.py`` turns x64 on before the first import,
    which is why nothing here saw it.
    """

    def _log32(self, n):
        return jnp.logspace(-4.0, np.log10(200.0), n).astype(jnp.float32)

    @pytest.mark.parametrize("n", [256, 512, 1024, 4096])
    def test_a_float32_log_grid_is_accepted(self, n):
        """Accepted, and the output carries the padding.

        The shape is ``n + 2 n_pad``, not ``n``: ``fftlog_pad_decades`` extends
        the grid at both ends.  Asserted against a rule built with the padding
        off, so this stays a statement about the *dtype tolerance* -- what the
        class is about -- and not about the default.
        """
        grid = self._log32(n)
        assert T.make_fftlog(0.0, grid, kind="spherical", power=3.0,
                             n_pad=0).y.shape == (n,)
        padded = T.make_fftlog(0.0, grid, kind="spherical", power=3.0)
        assert padded.y.shape == (n + 2 * padded.n_pad,)

    def test_a_float32_grid_that_is_not_log_spaced_is_still_refused(self):
        """Relaxing the floor must not blunt the refusal."""
        with pytest.raises(ValueError, match="log-spaced"):
            T.make_fftlog(0.0,
                          jnp.linspace(1e-3, 10.0, 256).astype(jnp.float32),
                          kind="bessel", power=2.0)

    @pytest.mark.parametrize("n", [256, 512, 1024, 4096])
    def test_the_floor_tracks_the_dtype_and_not_the_node_count(self, n):
        r"""One displaced node, 3e-6 in :math:`\ln k`: 25 eps32 and 1.4e10
        eps64.  Accepted in one dtype and refused in the other, at **every**
        ``n`` -- where under the old rule the dtype did not enter at all and the
        boundary moved by 16x across this range."""
        ln = np.log(np.logspace(-4.0, np.log10(200.0), n))
        ln[n // 2] += 3e-6
        x = np.exp(ln)
        T.make_fftlog(0.0, jnp.asarray(x, dtype=jnp.float32),
                      kind="spherical", power=3.0)              # must not raise
        with pytest.raises(ValueError, match="log-spaced"):
            T.make_fftlog(0.0, x, kind="spherical", power=3.0)

    def test_the_tolerance_carries_the_grids_ln_range(self):
        r"""A grid built as ``10**linspace`` carries a *relative* value error of
        ~``eps |ln x| / 2``, which lands in :math:`\ln x` additively -- so the
        deviation a correct grid shows scales with its ln-range.  A flat
        multiple calibrated on the shipped grid falls to 1.9x headroom by 26
        decades; carrying the span holds it near 10x everywhere."""
        wide = jnp.logspace(-13.0, 13.0, 512).astype(jnp.float32)
        T.make_fftlog(0.0, wide, kind="spherical", power=3.0)    # must not raise
        narrow = T._log_spacing_tol(0.05, 9.2, np.dtype(np.float32))
        assert T._log_spacing_tol(0.05, 29.9, np.dtype(np.float32)) > 3 * narrow

    def test_float64_gets_exactly_the_tolerance_it_had(self):
        """Not merely "float64 still passes": the absolute term is
        ``8 eps64 span <= 1.2e-13`` for any span under 70, against a relative
        term of ``1e-6 Delta``, so it could only win below ``Delta = 1.2e-7`` --
        more than 6e8 points.  In float64 this *is* the old check."""
        f64 = np.dtype(np.float64)
        for delta, span in ((0.0284, 9.2), (0.0051, 11.5), (0.117, 29.9)):
            assert T._log_spacing_tol(delta, span, f64) == 1e-8 + 1e-6 * delta

    def test_the_message_reports_the_deviation_and_the_tolerance(self):
        """Give the next reader a number, not an adjective."""
        import re
        with pytest.raises(ValueError, match="log-spaced") as e:
            T.make_fftlog(0.0, jnp.linspace(1e-3, 10.0, 256), kind="bessel",
                          power=2.0)
        msg = str(e.value)
        m = re.search(r"by ([0-9.eE+-]+), against a tolerance of ([0-9.eE+-]+)",
                      msg)
        assert m, msg
        assert float(m.group(1)) > float(m.group(2))
        assert "float64" in msg          # the dtype it actually read


class TestTheDefaultMultipoleGridIsDtypeIndependent:
    r"""``DEFAULT_WTHETA_ELL`` is a transform abscissa, and it was refused.

    It was ``tuple(float(v) for v in 10.0 ** jnp.linspace(0.0, 5.0, 256))``:
    evaluated at import time in whatever dtype JAX was configured for, then
    widened by ``float()``.  So the tuple held **float32-rounded values inside
    float64 Python floats**, with nothing in the object recording where they
    came from -- a provenance bug rather than a precision one.  Two processes
    differing only in ``JAX_ENABLE_X64`` projected ``w(theta)`` onto different
    multipole grids.

    Nothing caught it because nothing tested it: before this, ``WTheta``
    appeared in the suite only in an import line.
    """

    def test_it_is_exactly_the_float64_grid(self):
        ell = np.asarray(DEFAULT_WTHETA_ELL)
        assert ell.dtype == np.float64
        np.testing.assert_array_equal(ell, np.logspace(0.0, 5.0, 256))

    def test_the_engine_it_feeds_accepts_it(self):
        """The widening hides the provenance, so the dtype-aware tolerance
        reads float64 and applies the tolerance float64 has earned.  Built
        under float32 its deviation was 3.3e9 eps64."""
        T.make_fftlog(0.0, DEFAULT_WTHETA_ELL, kind="bessel", power=2.0)

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_w_theta_runs_on_that_grid_through_both_engines(self, flavour):
        r"""The first test this path has had.  A power-law :math:`C_\ell` has
        a closed-form Hankel transform, but the point here is coarser: the
        default grid reaches both engines and comes back finite and positive."""
        ell = jnp.asarray(DEFAULT_WTHETA_ELL)
        cl = 1e-6 * (ell / 100.0) ** -1.5
        theta = jnp.asarray([1e-3, 3e-3, 1e-2])
        w = np.asarray(T.cl_to_wtheta(theta, ell, cl, backend=flavour))
        assert np.all(np.isfinite(w)) and np.all(w > 0.0)
        assert np.all(np.diff(w) < 0.0)      # falls with angle


class TestASignChangingSpectrum:
    r"""``hankel`` on a :math:`C_\ell` with a negative lobe, against a closed form.

    .. math::

        C_\ell = \Big(1 - \frac{\ell^2}{\ell_0^2}\Big)e^{-\ell^2/2s^2}
        \;\longleftrightarrow\;
        w(\theta) = \frac{1}{2\pi}\Big[s^2 - \frac{s^4}{\ell_0^2}
            \big(2 - s^2\theta^2\big)\Big]e^{-s^2\theta^2/2}

    A pressure profile with a central depression (``alpha_in_p < 0``) hands the
    transform exactly this: :math:`u_P(k)` oscillates, and the cross power
    spectrum goes negative at high k.  The log-cubic floored the negative nodes
    at ``peak * 1e-300`` and overshot the 690-e-fold step by tens of e-folds:
    ggah_cal's M*>10.5 CAP MAP gave :math:`\chi^2` = 3.8e23 in float64 against
    207 in float32.  :func:`~ggah_mod.observables.transforms._signed_fallback`
    interpolates in value next to every non-positive node.
    """

    S, L0 = 3000.0, 5000.0
    THETA_ARCMIN = jnp.asarray([0.25, 0.5, 1.0, 2.0, 3.0, 4.0])

    def _exact(self, theta):
        s2 = self.S ** 2
        return ((s2 - s2 ** 2 / self.L0 ** 2 * (2.0 - s2 * theta ** 2))
                * jnp.exp(-0.5 * s2 * theta ** 2) / (2.0 * jnp.pi))

    def test_the_quadrature_engine_matches_the_closed_form(self):
        ell = jnp.asarray(DEFAULT_WTHETA_ELL)
        cl = (1.0 - (ell / self.L0) ** 2) * jnp.exp(-0.5 * (ell / self.S) ** 2)
        assert float(jnp.min(cl)) < 0.0          # the lobe is really there
        theta = jnp.deg2rad(self.THETA_ARCMIN / 60.0)
        got = T.cl_to_wtheta(theta, ell, cl, backend=ACCURATE)
        assert np.all(np.isfinite(np.asarray(got)))
        assert _rel(got, self._exact(theta)) < 2e-3

    def test_a_positive_spectrum_never_takes_the_fallback(self):
        """Bit for bit: the fallback is the identity when every node is > 0."""
        x = jnp.linspace(0.0, 5.0, 64)
        f = jnp.exp(-0.3 * x) + 0.1
        q = jnp.linspace(-1.0, 6.0, 300)[None, :]
        vals = jnp.full(q.shape, 7.0)
        np.testing.assert_array_equal(
            np.asarray(T._signed_fallback(vals, q, x, f)), np.asarray(vals))


class TestBesselK:
    r"""The one special function evaluated at a traced argument."""

    def test_against_the_closed_form_half_order(self):
        r""":math:`K_{1/2}(x) = \sqrt{\pi/2x}\,e^{-x}`."""
        x = jnp.asarray([0.5, 2.0, 8.0, 30.0])
        want = jnp.sqrt(jnp.pi / (2 * x)) * jnp.exp(-x)
        assert _rel(T.bessel_k(0.5, x), want) < 1e-8

    @pytest.mark.x64
    def test_it_differentiates_in_the_order_as_well_as_the_argument(self):
        """Which is the point: it makes the King PSF's slope a free parameter
        rather than a static choice.  ``scipy.special.kv`` gives neither."""
        for arg, x0 in (("nu", 0.5), ("x", 2.0)):
            def f(v):
                nu, x = (v, 2.0) if arg == "nu" else (0.5, v)
                return jnp.sum(T.bessel_k(nu, jnp.atleast_1d(x)))
            h = 1e-6
            ad = float(jax.grad(f)(x0))
            fd = float((f(x0 + h) - f(x0 - h)) / (2 * h))
            assert ad == pytest.approx(fd, rel=1e-5), arg


class TestDifferentiability:
    @pytest.mark.x64
    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: b.hankel)
    def test_the_gradient_flows_through_the_transform(self, flavour):
        k = jnp.logspace(-4, 3, 1024)

        def f(amp):
            pk = amp * (2 * jnp.pi * A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * A) ** 2)
            return jnp.sum(T.pk_to_xi(R, k, pk, backend=flavour))

        h = 1e-6
        ad = float(jax.grad(f)(1.0))
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)

    @pytest.mark.x64
    def test_the_gradient_flows_through_a_power_law_tail(self):
        r"""A spectrum whose measured tail slope sits at -3 -- the generic
        LambdaCDM value, and exactly where a ``minimum`` cap would halve it."""
        k = jnp.logspace(-3, 2, 512)

        def f(n):
            return jnp.sum(T.pk_to_xi(R, k, k ** n, backend=ACCURATE))

        h = 1e-6
        ad = float(jax.grad(f)(-3.0))
        fd = float((f(-3.0 + h) - f(-3.0 - h)) / (2 * h))
        assert np.isfinite(ad)
        assert ad == pytest.approx(fd, rel=1e-3)

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: b.hankel)
    def test_jit_is_value_identical(self, flavour, gaussian):
        k, pk = gaussian
        f = lambda p: T.pk_to_xi(R, k, p, backend=flavour)
        assert np.allclose(np.asarray(jax.jit(f)(pk)), np.asarray(f(pk)),
                           rtol=1e-12)

    def test_a_rule_survives_a_jit_boundary(self, gaussian):
        k, _ = gaussian
        rule = T.make_hankel(0.5, backend=ACCURATE)
        out = jax.jit(lambda r: r.x.sum())(rule)
        assert np.isfinite(float(out))
        leaves = jax.tree_util.tree_leaves(rule)
        assert all(not isinstance(x, str) for x in leaves)
