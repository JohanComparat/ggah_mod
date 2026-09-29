"""The differentiable P(k) backends, the neutrino table, and the CSST emulator.

The package needs four backends for two flavours: two Boltzmann solvers that are
accurate but not differentiable, and two differentiable ones that are less
accurate.  Without a differentiable P(k) the ``DIFFERENTIABLE`` flavour cannot be
constructed at all, so these are what make the two-flavour design real.
"""
import dataclasses
import importlib.util
import pathlib
import re

import ggah_mod
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE, ACCURATE
from ggah_mod.backend import (
    BACKENDS, HANKEL_ENGINES, TRACEABLE_CM,
    NEUTRINO_TWO_HALO, TWO_HALO_CONSISTENCY, TWO_HALO_SPECTRA, Backend,
)
from ggah_mod.cosmology import Cosmology, PLANCK18
from ggah_mod.cosmology.power import make_pk, PK_BACKENDS

from conftest import AnalyticPk
from ggah_mod.halos import (
    make_hmf, CONCENTRATION, make_concentration, c_bhattacharya13,
    c_diemer19, c_seppi21, parse_mass_def,
)

_K = np.logspace(-3, 1, 60)


# `TestNuRatioTable` stood here.  The table moved to `emu_pk`, and its own
# properties -- the exact identity at the LambdaCDM massless corner, C1
# continuity in each axis, the refusal to extrapolate -- are tested there,
# against the table directly.  What is tested on this side is the *seam*:
# `tests/test_pk_correction.py`.


class TestABackendWithoutNativeRedshift:
    """`AnalyticPk` is a test-only stub, and with the EH98 backend gone it is
    the only object declaring ``has_native_z = False``.  What is pinned here is
    not the stub but the *refusals* it triggers: a capability that is declared
    and then acted on, rather than inferred by calling the code and seeing
    whether it worked."""

    def test_declares_no_native_redshift(self):
        pk = AnalyticPk()
        assert pk.has_native_z is False and pk.differentiable is True

    def test_refuses_nonzero_redshift(self):
        """Rather than silently returning the z = 0 spectrum."""
        with pytest.raises(ValueError, match="no redshift dependence"):
            AnalyticPk().pk(_K, 1.0, PLANCK18)

    def test_growth_factor_refuses_it(self):
        from ggah_mod.cosmology.growth import growth_factor
        with pytest.raises(ValueError, match="has_native_z"):
            growth_factor(1.0, PLANCK18, AnalyticPk())

    # No accuracy test.  `AnalyticPk` makes no accuracy claim -- it exists to
    # be cheap and differentiable -- and comparing a test stub to CLASS would
    # be measuring nothing anyone relies on.

    @pytest.mark.x64
    def test_differentiable(self):
        pk = AnalyticPk()
        f = lambda v: jnp.log(pk.pk(jnp.array([0.1]), 0.0,
                                    PLANCK18.replace(ln10A_s=v))[0])
        assert float(jax.grad(f)(PLANCK18.ln10A_s)) == pytest.approx(1.0, rel=1e-6)


class TestEmuPk:
    @pytest.mark.slow
    def test_shape_against_class(self):
        a = np.asarray(make_pk("emu_pk").pk(_K, 0.0, PLANCK18))
        b = np.asarray(make_pk("class").pk(_K, 0.0, PLANCK18))
        r = (a / b) / np.median(a / b)
        assert np.max(np.abs(r - 1.0)) < 0.01

    def test_massless_total_and_cold_agree(self):
        """Learned agreement, not an identity, and the difference matters.

        The predecessor asserted *bit*-identity, and could: it built `pk_cb` as
        `pk` times a distilled ratio that was exactly 1 at the massless corner,
        so equality there was a property of the arithmetic.  Here the two are
        separate heads of one network, and at `sum_mnu = 0` they agree to 6e-4
        because the training set taught them to, not because anything forces
        it.  Asserting equality would be asserting something untrue; asserting
        nothing would drop the guarantee the class docstring claims, that two
        heads of one network cannot drift apart.  1e-3 is the band that says
        what is actually being promised.
        """
        cp = make_pk("emu_pk")
        c = Cosmology.create(sum_mnu=0.0)
        m = np.asarray(cp.pk(_K, 0.0, c))
        cb = np.asarray(cp.pk_cb(_K, 0.0, c))
        assert np.max(np.abs(m / cb - 1.0)) < 1e-3

    def test_training_box_is_enforced(self):
        cp = make_pk("emu_pk")
        with pytest.raises(ValueError, match="training box"):
            cp.pk(_K, 0.0, PLANCK18.replace(h=0.95))

    def test_training_box_can_be_waived(self):
        from ggah_mod.cosmology.power import GgahEmuPk
        assert np.all(np.isfinite(
            np.asarray(GgahEmuPk(check_box=False).pk(_K, 0.0,
                                                     PLANCK18.replace(h=0.95)))))

    @pytest.mark.x64
    @pytest.mark.parametrize("name", ["ln10A_s", "Omega_m", "h", "n_s", "sum_mnu"])
    def test_gradients_match_finite_differences(self, name):
        cp = make_pk("emu_pk")
        f = lambda v: jnp.log(cp.pk(jnp.array([0.1]), 0.0,
                                    PLANCK18.replace(**{name: v}))[0])
        x = float(getattr(PLANCK18, name))
        step = 1e-6 * max(abs(x), 1e-3)
        assert float(jax.grad(f)(x)) == pytest.approx(
            float((f(x + step) - f(x - step)) / (2 * step)), rel=3e-3, abs=1e-9)

    @pytest.mark.x64
    @pytest.mark.parametrize("name", ["w0", "wa"])
    def test_the_spectrum_responds_to_dark_energy(self, name):
        """w0 and wa are network inputs, so this is cheap -- and it is kept
        because of what it caught when they were not.

        The predecessor network's training box was LambdaCDM, so the spectrum
        could not respond to either parameter at all, and an
        autodiff-versus-finite-difference check *passes* on a derivative that
        is absent: both estimators agree, correctly, that it is zero.  No
        within-flavour check could see it.  It showed up only against a
        Boltzmann backend, where `derivative_parity` reported 100 per cent for
        every spectrum-derived quantity in these two directions.

        The consequence was not "DIFFERENTIABLE is a bit worse here".  The
        background is shared analytic code and always *did* respond to w0 and
        wa, so a Fisher forecast got dark-energy sensitivity from distances,
        none from the spectrum, and no error.  A non-zero assertion is the
        cheapest thing that would have failed, so it stays even though the
        spectrum that needed it is gone.
        """
        cp = make_pk("emu_pk")
        f = lambda v: jnp.sum(jnp.log(
            cp.pk(jnp.asarray([0.05, 0.1, 0.5]), 0.0,
                  PLANCK18.replace(**{name: v}))))
        got = float(jax.grad(f)(float(getattr(PLANCK18, name))))
        assert got != 0.0, f"dlnP/d{name} is still absent"
        assert abs(got) > 1e-4, f"dlnP/d{name} = {got:.3e} is implausibly small"

    @pytest.mark.x64
    @pytest.mark.parametrize("name", ["w0", "wa"])
    def test_the_background_does_respond_to_dark_energy(self, name):
        """The other half of the statement above, and the reason it is subtle."""
        from ggah_mod.cosmology import hubble_e
        f = lambda v: jnp.log(hubble_e(1.0, PLANCK18.replace(**{name: v})))
        assert abs(float(jax.grad(f)(float(getattr(PLANCK18, name))))) > 1e-3

    def test_jit(self):
        cp = make_pk("emu_pk")
        g = jax.jit(lambda c: cp.pk(jnp.array([0.1]), 0.0, c)[0])
        assert float(g(PLANCK18)) > 0.0

    def test_validation_does_not_break_tracing(self):
        """The box check calls float(); under a trace that raises
        ConcretizationTypeError and kills the gradient, so it must be skipped
        rather than attempted."""
        cp = make_pk("emu_pk")
        f = lambda v: cp.pk(jnp.array([0.1]), 0.0, PLANCK18.replace(h=v))[0]
        assert np.isfinite(float(jax.grad(f)(PLANCK18.h)))


class TestFlavoursAreConstructible:
    def test_fast_backend_has_a_differentiable_pk(self):
        """Without one, DIFFERENTIABLE is a declaration with nothing behind it."""
        assert DIFFERENTIABLE.pk in PK_BACKENDS
        assert make_pk(DIFFERENTIABLE.pk).differentiable is True

    def test_accurate_backend_pk_exists(self):
        assert make_pk(ACCURATE.pk).differentiable is False

    @pytest.mark.parametrize("name", sorted(BACKENDS))
    def test_every_declared_name_resolves(self, name):
        """A backend's ``cm_model`` and ``mdef`` must name things that exist.

        Both were wrong and neither showed it. ``ACCURATE`` asked for
        ``"diemer19"`` and ``TRACEABLE_CM`` offered ``"diemer19_jax"``; the
        registry calls that relation ``"diemer19"``, so neither name resolved.
        Nothing consumed ``cm_model`` yet, so nothing raised -- layer 3 would
        have, on its first call.

        The root cause was a check that validated one declaration against
        another: ``_validate`` compared ``cm_model`` to ``TRACEABLE_CM``, and
        ``TRACEABLE_CM`` was compared to nothing. This test is the missing end
        of that chain.
        """
        b = BACKENDS[name]
        assert b.cm_model in CONCENTRATION, (
            f"{name}.cm_model = {b.cm_model!r} is not in the registry")
        parse_mass_def(b.mdef)          # raises if it does not parse

    def test_traceable_cm_names_exist(self):
        """The set of pure-JAX relations cannot name one that is not there."""
        missing = sorted(set(TRACEABLE_CM) - set(CONCENTRATION))
        assert not missing, f"TRACEABLE_CM names non-existent relations: {missing}"

    @pytest.mark.parametrize("name", sorted(TRACEABLE_CM))
    def test_traceable_cm_really_is_traceable(self, name):
        """Declared differentiable, then differentiated -- not taken on trust.

        The empirical relations legitimately return a zero gradient with respect
        to the cosmology, because they cannot see one; what is asserted here is
        that a gradient runs at all and is finite, which is what ``traced``
        promises.
        """
        m = jnp.asarray([1e13])
        s = jnp.asarray([1.5])
        # Keyed by name, not an if/elif ending in `else`.  It used to end in
        # one, so a relation added to TRACEABLE_CM and to nothing else was
        # differentiated as `diemer19` a second time and reported as a pass.
        cases = {
            "duffy08": lambda x: jnp.sum(
                make_concentration("duffy08")(m * x, 0.0, "200c")),
            "dutton14": lambda x: jnp.sum(
                make_concentration("dutton14")(m * x, 0.0, "200c")),
            "klypin16": lambda x: jnp.sum(
                make_concentration("klypin16")(m * x, 0.0, "200c")),
            "bhattacharya13": lambda x: jnp.sum(c_bhattacharya13(s * x, 1.0, "200c")),
            "diemer19": lambda x: jnp.sum(c_diemer19(s * x, -2.0, 0.52)),
            "seppi21": lambda x: jnp.sum(c_seppi21(s * x, 0.5, "vir")),
        }
        assert name in cases, (
            f"{name!r} is in TRACEABLE_CM but this test has no case for it, so "
            f"nothing here would differentiate it")
        g = float(jax.grad(cases[name])(1.0))
        assert np.isfinite(g)

    def test_a_bad_name_is_refused_at_construction(self):
        """Not deferred to first use, where the layer above would carry it."""
        with pytest.raises(ValueError, match="concentration relation"):
            ACCURATE.with_(cm_model="no_such_relation")
        with pytest.raises(ValueError, match="mass definition"):
            ACCURATE.with_(mdef="500q")


#: CSSTemu is optional and installed from git.  Checked with `find_spec` rather
#: than an import, because CEmulator imports only after the shim has run.
NO_CSSTEMU = importlib.util.find_spec("CEmulator") is None


@pytest.mark.slow
@pytest.mark.skipif(NO_CSSTEMU, reason="CSSTemu (CEmulator) is not installed")
class TestCsstMassFunction:
    def test_runs_and_is_positive(self):
        hmf = make_hmf("csst")
        m = np.logspace(11, 15, 9)
        n = np.asarray(hmf.dndm(m, 0.0, PLANCK18))
        assert np.all(n > 0) and np.all(np.diff(n) < 0)

    def test_declares_itself_non_differentiable(self):
        """It is a scikit-learn GP behind numpy, so DIFFERENTIABLE cannot use it."""
        assert make_hmf("csst").differentiable is False

    def test_refuses_untrained_redshift(self):
        with pytest.raises(ValueError, match="trained range"):
            make_hmf("csst").dndm(np.logspace(12, 14, 3), 5.0, PLANCK18)

    def test_rejects_unknown_mass_definition(self):
        with pytest.raises(ValueError, match="massdef"):
            make_hmf("csst", massdef="FoF200")

    def test_agrees_with_the_fitting_function_to_the_expected_level(self):
        """A simulation emulator and a fit to simulations should agree at the
        ~10% level over the well-sampled range; that they do not agree better
        is the reason the emulator exists."""
        m = np.logspace(12, 14.5, 8)
        a = np.asarray(make_hmf("csst").dndm(m, 0.0, PLANCK18))
        b = np.asarray(make_hmf("tinker08", pk=make_pk("class")).dndm(m, 0.0, PLANCK18))
        assert np.max(np.abs(a / b - 1.0)) < 0.15

    def test_dispatch(self):
        with pytest.raises(ValueError, match="needs a linear-P"):
            make_hmf("tinker08")
        with pytest.raises(ValueError, match="unknown mass function"):
            make_hmf("nonesuch")


class TestEveryBackendFieldHasAReader:
    r"""A declared knob nothing consumes is a knob nothing checks.

    This package has been bitten by it twice.  ``ACCURATE.cm_model`` asked for
    ``"diemer19"`` and ``TRACEABLE_CM`` offered ``"diemer19_jax"``, and the
    registry has neither -- unnoticed until layer 3 became the first consumer.
    Then ``gas_ft`` and ``n_gl``: ``DIFFERENTIABLE`` declared ``"surrogate"`` with
    ``n_gl = 64``, ``_validate`` *refused* anything else on a traced backend on
    the grounds that ``"quadrature"`` was "the numpy quadrature path", and
    ``HotGasDPM`` hard-coded 128 and always took that path.  Three statements,
    all false, held together by the fact that nothing read the field.

    So the rule is inverted here: a field is read, or it is on the list below
    with the milestone that will read it.  The list shrinks; it must never grow
    without a reason next to the entry.
    """

    #: Fields no module reads yet, and what will.  Entries are removed as the
    #: layers land -- an entry that outlives its milestone is a dead knob.
    #: **Empty**, and that is the milestone.
    #:
    #: It has had entries at every earlier stage -- ``two_halo_spectrum`` until
    #: layer 4 read it, the five quadrature sizes until layer 5 did, and
    #: ``n_z_proj`` until the Limber grid did.  Each was removed by the test
    #: below failing, which is the mechanism working: a knob that acquires a
    #: reader must lose its exemption in the same commit, or the exemption
    #: outlives the reason for it.
    #:
    #: A new entry is legitimate only for a field added *ahead* of the layer
    #: that will read it, and it must name that layer.
    UNREAD: dict[str, str] = {}

    #: Names a backend is bound to at its call sites.
    #:
    #: Searching for a bare ``.n_k`` would match ``HaloField.n_k``, which is a
    #: different quantity with the same spelling -- exactly the confusion this
    #: file is about.  The bare ``b.`` form is real (``make_field`` writes
    #: ``b = resolve_backend(backend)``) but far too generic on its own: it
    #: matched ``b.name`` in ``spectra/pk.py``, where ``b`` is a *tracer* label.
    #: So it counts only in a file that resolved a backend in the first place.
    _EXPLICIT = r"\b(?:backend|_backend)\."
    _LOCAL = r"\bb\."

    @staticmethod
    def _sources():
        root = pathlib.Path(ggah_mod.__file__).parent
        return {p: p.read_text() for p in root.rglob("*.py")
                if p.name != "backend.py"}

    def _readers(self, field: str) -> list[str]:
        explicit = re.compile(self._EXPLICIT + field + r"\b")
        local = re.compile(self._LOCAL + field + r"\b")
        out = []
        for path, text in self._sources().items():
            if explicit.search(text) or (
                    "resolve_backend" in text and local.search(text)):
                out.append(path.name)
        return out

    def _is_read(self, field: str) -> bool:
        return bool(self._readers(field))

    @pytest.mark.parametrize(
        "field", [f.name for f in dataclasses.fields(Backend)])
    def test_the_field_is_read_or_declared_unread(self, field):
        if field in self.UNREAD:
            pytest.skip(f"not yet consumed -- {self.UNREAD[field]}")
        if field in ("name", "traced"):
            return                      # read by _validate and by __str__
        assert self._is_read(field), (
            f"Backend.{field} is declared and validated but no module reads "
            f"it.  Either wire it up, or add it to UNREAD with the milestone "
            f"that will -- a knob nothing consumes is a knob nothing checks, "
            f"and cm_model and gas_ft were both wrong for exactly that reason.")

    def test_the_unread_list_has_no_dead_entries(self):
        """A field that *is* read must not still be listed as pending."""
        stale = {f: self._readers(f) for f in self.UNREAD if self._is_read(f)}
        assert not stale, (
            f"these are listed as not-yet-consumed but something reads them: "
            f"{stale}.  Remove them from UNREAD -- a stale entry is how a knob "
            f"stops being checked again.")

    def test_the_unread_list_names_only_real_fields(self):
        names = {f.name for f in dataclasses.fields(Backend)}
        assert set(self.UNREAD) <= names


class TestTheLayerFourAndFiveKnobs:
    """The fields added for the spectra and observables layers."""

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE])
    def test_every_declared_name_resolves(self, flavour):
        assert flavour.hankel in HANKEL_ENGINES
        assert flavour.two_halo_spectrum in TWO_HALO_SPECTRA
        assert flavour.two_halo_consistency in TWO_HALO_CONSISTENCY
        assert flavour.neutrino_two_halo in NEUTRINO_TWO_HALO

    @pytest.mark.parametrize("field,bad", [
        ("hankel", "ogata"), ("two_halo_spectrum", "linear"),
        ("two_halo_consistency", "yes"), ("neutrino_two_halo", "halo"),
    ])
    def test_a_name_outside_its_registry_is_refused(self, field, bad):
        with pytest.raises(ValueError, match="does not exist"):
            DIFFERENTIABLE.with_(**{field: bad})

    def test_the_gas_ft_field_is_gone(self):
        """`DIFFERENTIABLE` declared `"surrogate"`, nothing implemented it, and nothing
        could: the DPM shape is a gNFW with free slopes, whose transform has no
        closed form, so `gnfw_uk` is itself the same quadrature.  A field with
        one legal value declares no choice."""
        assert not hasattr(DIFFERENTIABLE, "gas_ft")
        assert not any(f.name == "gas_ft" for f in dataclasses.fields(Backend))

    def test_the_accurate_flavour_integrates_the_gas_more_finely(self):
        """What `gas_ft` was gesturing at, and what actually differs."""
        assert ACCURATE.n_gl > DIFFERENTIABLE.n_gl

    def test_a_zero_projection_grid_is_refused(self):
        with pytest.raises(ValueError, match="n_z_proj"):
            DIFFERENTIABLE.with_(n_z_proj=1)

    def test_a_traced_backend_refuses_a_non_differentiable_spectrum(self):
        """`Backend.pk` was declared, validated against TRACEABLE_PK, and read
        by no module -- so a DIFFERENTIABLE field built on CambPk passed every check and
        would have produced a gradient with dP/dtheta silently missing.  That
        is the exact failure `traced` exists to prevent."""
        from ggah_mod.halos.field import make_field

        class NotDifferentiable:
            name, has_native_z, differentiable = "camb", True, False

        with pytest.raises(ValueError, match="silently omit"):
            make_field(PLANCK18, DIFFERENTIABLE, NotDifferentiable(), z=0.0)

    def test_the_missing_spectrum_message_names_the_declared_one(self):
        """The message has to name the flavour's *own* backend.

        Matched against ``DIFFERENTIABLE.pk`` rather than against a literal:
        a test that spells the current default is a test that fails when the
        default moves, which says nothing about whether the message is right.
        """
        from ggah_mod.halos.field import make_field
        with pytest.raises(ValueError,
                           match=rf"make_pk\('{DIFFERENTIABLE.pk}'\)"):
            make_field(PLANCK18, DIFFERENTIABLE, None)


class TestTheResolverRefusesAndTheRetiredNamesWarn:
    """Two small contracts the suite exercised nowhere.

    Both are about names: one a backend that does not exist, one a backend
    that used to.  Neither is expensive to get wrong -- which is exactly why
    nothing had checked them.
    """

    def test_an_unknown_backend_name_is_refused_and_lists_the_real_ones(self):
        from ggah_mod.backend import BACKENDS, resolve_backend

        with pytest.raises(ValueError, match="unknown backend") as e:
            resolve_backend("fast-ish")
        # The message has to carry the alternatives, or the caller is left
        # guessing at the spelling of something they have never seen.
        for name in BACKENDS:
            assert name in str(e.value)

    @pytest.mark.parametrize("old,new", [("FAST", "DIFFERENTIABLE"),
                                         ("REFERENCE", "ACCURATE")])
    def test_a_retired_flavour_name_still_resolves_but_warns(self, old, new):
        """Retired, not removed: `0a9072e` kept both as warning aliases.

        A rename that breaks every caller silently is a worse trade than one
        that keeps working and says so, and the warning is the whole of the
        difference -- so it is the part worth pinning.
        """
        import ggah_mod.backend as B

        with pytest.warns(DeprecationWarning, match=new):
            got = getattr(B, old)
        assert got is getattr(B, new)

    def test_a_name_that_was_never_a_flavour_still_raises_AttributeError(self):
        """The module `__getattr__` must not swallow ordinary typos."""
        import ggah_mod.backend as B

        with pytest.raises(AttributeError, match="has no attribute"):
            B.SLOW


class TestCsstHmfDoesNotDependOnImportOrder:
    r"""`CsstHMF()` has to work in a process that has imported nothing else.

    It did not.  `__init__` imported `CEmulator` *before* the shim, and the
    shim is what restores `scipy.integrate.simps` -- which CEmulator imports at
    its own module scope.  So the class worked only when some earlier import in
    the same process had already pulled the shim in, and the whole
    `TestCsstMassFunction` class above passed for that reason rather than on
    its own merits: running this file alone gave five ImportErrors.

    Worse, the handler reported it as "CSSTemu is not installed" about a
    package that was installed and merely needed the shim first -- a message
    that sends the reader to fix the wrong thing.

    A subprocess is the only honest way to check this.  Inside an already-warm
    interpreter the bug is invisible by construction, which is exactly how it
    survived.
    """

    def test_it_constructs_in_a_fresh_interpreter(self):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-c",
             "from ggah_mod.halos.mass_function import CsstHMF;"
             " h = CsstHMF(); print('ok', h.patched)"],
            capture_output=True, text=True, timeout=600)
        if "could not be imported" in r.stderr or "not installed" in r.stderr:
            pytest.skip("CSSTemu is genuinely absent from this environment")
        assert r.returncode == 0, r.stderr[-2000:]
        assert r.stdout.startswith("ok"), r.stdout
