r"""Parity: the ported occupation kernels against `hod_mod`, at the same parameters.

These models are *transcriptions*.  Nothing about them is meant to be new, so
the only interesting question is whether the transcription is faithful, and the
answer should be "to round-off" rather than "to a tolerance".

Where this suite disagrees with the main one
--------------------------------------------

Two deliberate departures, each asserted here rather than left for a reader to
discover:

* **van Uitert satellites.**  The predecessor's docstring writes the modified
  Schechter with a ``dM_*/M_*`` measure and its code integrates ``u^(alpha+1)``,
  which is the ``dM_*`` measure.  The code is right -- that is what the
  published parameters were fitted through -- so the port follows the code and
  the docstring here says so.
* **Zacharegkas satellites** take no ``width_logmstar`` or ``f_cen``.  The
  predecessor's signature omits them too; this records that the omission is the
  model, not an oversight.

Comparisons use **keyword arguments throughout**.  Positionally, several of
these functions differ only by an adjacent transposition -- ``alpha_sat`` sits
before ``log10m_sat`` in one and after it in another, and ``log10m_star0``
before ``log10m_h1`` in a third.  Getting one wrong produces a number 46 orders
of magnitude out, which is obvious, or a few per cent out, which is not.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from ggah_mod.sectors import occupation as O

pytestmark = pytest.mark.parity

hod = pytest.importorskip(
    "hod_mod.connection.hod",
    reason="hod_mod is a test-only dependency; install it to run parity")

from hod_mod.connection.hod import base as HB           # noqa: E402
from hod_mod.connection.hod import guo as HG            # noqa: E402
from hod_mod.connection.hod import kravtsov04 as HK     # noqa: E402
from hod_mod.connection.hod import leauthaud12 as HL    # noqa: E402
from hod_mod.connection.hod import more15 as HM         # noqa: E402
from hod_mod.connection.hod import vanuitert16 as HV    # noqa: E402
from hod_mod.connection.hod import zacharegkas25 as HZC  # noqa: E402
from hod_mod.connection.hod import zumandelbaum15 as HZ  # noqa: E402

LOG10M = jnp.asarray(np.linspace(10.5, 15.5, 21))

#: Compared as ``|a-b| / peak(b)`` rather than element-wise relative.
#:
#: The occupations span 300 decades: at the low-mass end ``n_cen`` underflows
#: toward 1e-300 and ``exp(-M_cut/M)`` toward 1e-14, and an element-wise
#: relative difference between two denormals is noise with no bearing on
#: whether the transcription is right.  Peak-normalised, a real error anywhere
#: that matters still shows.
ATOL_ON_PEAK = 1e-12


def _same(got, want, tag=""):
    got, want = np.asarray(got), np.asarray(want)
    peak = max(float(np.max(np.abs(want))), 1e-300)
    assert np.max(np.abs(got - want)) / peak < ATOL_ON_PEAK, (
        f"{tag}: max |diff|/peak = {np.max(np.abs(got - want)) / peak:.3e}")


#: ``ggah_mod`` name -> ``hod_mod`` name, where the two packages disagree.
#:
#: PLAN.md item **G5** renamed two of this package's scatter parameters, because
#: ``leauthaud12`` was spelling a scatter in ``log10 M*`` with the *halo*-mass
#: name it shared with ``zheng07``, and the erf-width family differed from the
#: Gaussian one by a single underscore.  The predecessor still uses the old
#: spellings.
#:
#: Keeping a translation here rather than reverting the rename is the point of a
#: parity suite: the numbers must agree, the *names* need not, and a package that
#: matched its predecessor's names in order to make a test pass would have
#: imported the defect along with them.
_HOD_MOD_NAMES = {
    "sigma_logmstar": "sigma_logm",       # hod_mod's leauthaud12 keeps the collision
    "width_logmstar": "sigma_logm_star",
    "alpha_shmr_k18": "alpha_shmr",     # hod_mod spells both SHMRs' slope alike
}


def _for_hod_mod(params: dict) -> dict:
    """The same values under the predecessor's spelling."""
    return {_HOD_MOD_NAMES.get(k, k): v for k, v in params.items()}


def _sub(name, *keys):
    d = O.DEFAULTS[name]
    return {k: d[k] for k in keys}


class TestZhengFamily:
    def test_n_cen(self):
        p = _sub("zheng07", "log10mmin", "sigma_logm")
        _same(O.n_cen_zheng07(LOG10M, **p), HB.n_cen(LOG10M, **p))

    def test_n_sat(self):
        p = O.DEFAULTS["zheng07"]
        _same(O.n_sat_zheng07(LOG10M, **p), HB.n_sat(LOG10M, **p))

    def test_kravtsov04_n_sat(self):
        p = O.DEFAULTS["kravtsov04"]
        _same(O.n_sat_kravtsov04(LOG10M, **p),
              HK.n_sat_kravtsov04(LOG10M, **p))

    def test_the_two_satellite_cutoffs_really_differ(self):
        """Or the two tests above are checking one function twice.

        Compared **at the cutoff**, not over the whole range: far above it both
        tend to the same power law and agree to 0.3%, so a peak-normalised
        comparison would call two genuinely different models the same.  At
        `M = M_0` the distinction is exact -- Zheng's bracket vanishes, and
        Kravtsov's exponential does not.
        """
        p = O.DEFAULTS["kravtsov04"]
        at_m0 = jnp.asarray(p["log10m0"])
        assert float(O.n_sat_zheng07(at_m0, **p)) == 0.0
        assert float(O.n_sat_kravtsov04(at_m0, **p)) > 0.0


class TestMore15:
    def test_n_cen(self):
        p = _sub("more15", "log10mmin", "sigma_logm", "alpha_inc", "log10m_inc")
        _same(O.n_cen_more15(LOG10M, **p), HM.n_cen_more15(LOG10M, **p))

    def test_n_sat(self):
        p = O.DEFAULTS["more15"]
        _same(O.n_sat_more15(LOG10M, **p), HM.n_sat_more15(LOG10M, **p))

    def test_const_finc_n_cen(self):
        p = _sub("more15_const", "log10mmin", "sigma_logm", "f_inc")
        _same(O.n_cen_more15_const(LOG10M, **p),
              HM.n_cen_more15_const_finc(LOG10M, **p))

    def test_const_finc_n_sat(self):
        p = O.DEFAULTS["more15_const"]
        _same(O.n_sat_more15_const(LOG10M, **p),
              HM.n_sat_more15_const_finc(LOG10M, **p))


class TestGuo:
    CEN = ("log10m_star0", "log10m1_shmr", "alpha_shmr", "beta_shmr",
           "width_logmstar", "f_cen", "log10m_star_min_cen", "sigma_c_cen")
    SAT = ("log10m_star0", "log10m1_shmr", "alpha_shmr", "beta_shmr",
           "f_sat", "log10m_star_min_sat", "sigma_c_sat", "log10m1_sat",
           "alpha_sat")

    def test_shmr(self):
        p = _sub("guo18", "log10m_star0", "log10m1_shmr", "alpha_shmr",
                 "beta_shmr")
        # The predecessor names the same four `log10m1`, `alpha`, `beta`.  Here
        # they carry `_shmr` suffixes, because `log10m1` and `alpha` are also
        # the names of *satellite* parameters in the sibling models and the
        # bare names are what let a fit configuration pass one where the other
        # belongs.
        _same(O.shmr_guo18(LOG10M, **p),
              HG.shmr_guo18(LOG10M, log10m_star0=p["log10m_star0"],
                            log10m1=p["log10m1_shmr"],
                            alpha=p["alpha_shmr"], beta=p["beta_shmr"]))

    def test_n_cen(self):
        p = _sub("guo18", *self.CEN)
        _same(O.n_cen_guo18(LOG10M, **p),
              HG.n_cen_guo18(LOG10M, **_for_hod_mod(p)))

    def test_n_sat(self):
        p = _sub("guo18", *self.SAT)
        _same(O.n_sat_guo18(LOG10M, **p),
              HG.n_sat_guo18(LOG10M, sigma_logm_star=O.DEFAULTS["guo18"]
                             ["width_logmstar"], **_for_hod_mod(p)))

    def test_guo19_quenching_suppresses_massive_halos(self):
        """The direction is the whole point of the ELG variant."""
        fq = np.asarray(O.quenched_fraction_guo19(LOG10M, 12.0))
        assert np.all(np.diff(fq) > 0) and fq[0] < 0.1 < 0.9 < fq[-1]


class TestZuMandelbaum15:
    CEN = ("log10m_star_thresh", "lg_m1h", "lg_m0star", "beta", "delta",
           "gamma", "sigma_lnmstar", "eta", "fc")

    def test_n_cen(self):
        p = _sub("zumandelbaum15", *self.CEN)
        _same(O.n_cen_zu15(LOG10M, **p), HZ.n_cen_thresh_zu15(LOG10M, **p))

    def test_n_sat(self):
        p = O.DEFAULTS["zumandelbaum15"]
        _same(O.n_sat_zu15(LOG10M, **p), HZ.n_sat_thresh_zu15(LOG10M, **p))


class TestLeauthaud12:
    CEN = ("log10m_star_thresh", "log10m1", "log10m_star0", "beta", "delta",
           "gamma", "sigma_logmstar")

    def test_n_cen(self):
        p = _sub("leauthaud12", *self.CEN)
        _same(O.n_cen_leauthaud12(LOG10M, **p),
              HL.n_cen_leauthaud12(LOG10M, **_for_hod_mod(p)))

    def test_n_sat(self):
        p = O.DEFAULTS["leauthaud12"]
        _same(O.n_sat_leauthaud12(LOG10M, **p),
              HL.n_sat_leauthaud12(LOG10M, **_for_hod_mod(p)))


class TestZacharegkas25:
    CEN = ("log10m_star_thresh", "log10m1_shmr", "log10eps", "alpha_shmr_k18",
           "gamma_shmr", "delta_shmr", "width_logmstar", "f_cen")

    def test_n_cen(self):
        p = _sub("zacharegkas25", *self.CEN)
        _same(O.n_cen_zacharegkas25(LOG10M, **p),
              HZC.n_cen_thresh_z25(LOG10M, **_for_hod_mod(p)))

    def test_n_sat(self):
        p = O.DEFAULTS["zacharegkas25"]
        p = {k: v for k, v in p.items()
             if k not in ("width_logmstar", "f_cen")}
        _same(O.n_sat_zacharegkas25(LOG10M, **p),
              HZC.n_sat_thresh_z25(LOG10M, **_for_hod_mod(p)))

    def test_the_satellite_term_takes_no_central_parameters(self):
        """Recorded: it is not gated by the central occupation, unlike the
        Zheng, More and Zu & Mandelbaum forms."""
        import inspect
        got = set(inspect.signature(O.n_sat_zacharegkas25).parameters)
        assert "f_cen" not in got and "width_logmstar" not in got


class TestVanUitert16:
    SHARED = ("log10m_star_lo", "log10m_star_hi", "log10m_star0", "log10m_h1",
              "beta1", "log10_beta2")

    def test_n_cen(self):
        p = _sub("vanuitert16", *self.SHARED, "sigma_c")
        _same(O.n_cen_vanuitert16(LOG10M, **p),
              HV.n_cen_vanuitert16(LOG10M, **p))

    def test_n_sat_follows_the_code_not_the_docstring(self):
        """The `dM_*` measure, i.e. `u^(alpha_s+1)`.

        Dropping the Jacobian is a factor of order `u` inside the integral,
        which varies across the stellar-mass bin -- so it is *not* absorbable
        into the amplitude `phi_s`, and a fit would not hide it.
        """
        p = _sub("vanuitert16", *self.SHARED, "alpha_s", "b0", "b1")
        _same(O.n_sat_vanuitert16(LOG10M, **p),
              HV.n_sat_vanuitert16(LOG10M, **p))
