"""A compatibility shim for CSSTemu under numpy >= 2.

Two CSSTemu methods -- ``HMFbase_gp.get_data`` and
``PkcbLin_gp.get_pkcbLin`` -- do::

    ypred = np.zeros((self.nvec))
    for ivec in range(self.nvec):
        ypred[ivec] = self._GPR[ivec].predict(Normcosmo)

``predict`` returns an array of shape ``(1,)`` because ``Normcosmo`` is
``(1, n_params)``.  Assigning a size-1 array into a scalar slot was deprecated
in numpy 1.25 and is an error from numpy 2::

    ValueError: setting an array element with a sequence.

So on numpy >= 2 the CSST mass function raises before returning anything.  Both
must be patched, not just the first: the mass function reaches the second
internally, through ``get_dndlnM -> get_dndlnM_Castro23 -> _get_dlns_dlnR ->
get_sigma_cb_z``.

The fix is one call to :func:`numpy.ravel`, upstream; until that lands, this
patches the methods in place.

Deliberately narrow:

* applied only when the failure is real -- it is *probed*, by calling the method
  and catching the error, rather than assumed from a version number, so a fixed
  upstream release stops being patched automatically;
* idempotent, and it records what it did on the class so a second import is a
  no-op;
* the replacement is the original method with ``np.ravel(...)[0]`` added and
  nothing else changed.

Patching a third-party library is not a thing to do lightly, and the alternative
here -- silently dropping the most accurate mass function in the package -- is
worse.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ensure_cemulator_works"]

_FLAG = "_ggah_numpy2_patched"

#: numpy names CSSTemu uses that numpy 2 removed.  Each is a pure rename with
#: identical semantics, so restoring the old spelling is safe: it can only make
#: previously-failing code run, never change what working code does.
_REMOVED_NUMPY_ALIASES = {
    "trapz": "trapezoid",     # np.trapz -> np.trapezoid  (5 uses)
    "NaN": "nan",             # np.NaN   -> np.nan        (4 uses)
}


#: scipy names CSSTemu uses that scipy removed, same rule as above.  ``simps``
#: was renamed ``simpson`` and deprecated for four releases before removal, so
#: the two are the same function under two spellings.  It is reached on
#: ``import CEmulator``, before any object exists to patch a method on, which is
#: why it belongs beside the numpy aliases rather than in
#: :func:`ensure_cemulator_works`.
_REMOVED_SCIPY_ALIASES = {
    "simps": "simpson",       # scipy.integrate.simps -> simpson
}


def _restore_scipy_aliases() -> list[str]:
    """Put back the ``scipy.integrate`` names CSSTemu was written against.

    Same justification as :func:`_restore_numpy_aliases`, and the same narrow
    scope: an exact rename, restored only when it is absent, so it can make
    failing code run and cannot change what working code does.
    """
    from scipy import integrate as _integrate

    restored = []
    for old, new in _REMOVED_SCIPY_ALIASES.items():
        if not hasattr(_integrate, old) and hasattr(_integrate, new):
            setattr(_integrate, old, getattr(_integrate, new))
            restored.append(f"scipy.integrate.{old} -> {new}")
    return restored


#: Restored at *import* of this module, not inside
#: :func:`ensure_cemulator_works`.  The numpy aliases are reached when a method
#: runs, so patching them beside the methods is early enough; ``simps`` is
#: reached by ``import CEmulator`` itself, and every caller imports this module
#: on the line before it -- so this is the last moment that still works.
_SCIPY_RESTORED = _restore_scipy_aliases()


def _restore_numpy_aliases() -> list[str]:
    """Put back the numpy names CSSTemu was written against.

    Deeper than patching a method, and worth being explicit about: this mutates
    the ``numpy`` module for the whole process.  It is justified because each
    name is an exact alias of one that still exists -- adding ``np.trapz`` back
    as ``np.trapezoid`` cannot alter the behaviour of any code that does not
    already reference ``np.trapz``, and all such code is currently broken.

    The alternative was patching five call sites across three CSSTemu modules,
    which is more code and more likely to drift from upstream.
    """
    restored = []
    for old, new in _REMOVED_NUMPY_ALIASES.items():
        if not hasattr(np, old) and hasattr(np, new):
            setattr(np, old, getattr(np, new))
            restored.append(f"np.{old} -> np.{new}")
    return restored


def _fixed_get_data(self):
    """CSSTemu ``HMFbase_gp.get_data``, with the size-1 assignment fixed."""
    if self.NormBeforeGP:
        Normcosmo = self.paramSS.transform(self.ncosmo)
    else:
        Normcosmo = np.copy(self.ncosmo)
    ypred = np.zeros((self.nvec))
    for ivec in range(self.nvec):
        # The only change: predict() returns shape (1,), not a scalar.
        ypred[ivec] = np.ravel(self._GPR[ivec].predict(Normcosmo))[0]
    if self.NormBeforeGP:
        ypred = self.pkcoeffSS.inverse_transform(ypred.reshape(1, -1))[0]
    ypred = (ypred @ self._PCA_components) + self._PCA_mean
    if self.NormBeforePCA:
        ypred = ypred * self.pcaSS_scale + self.pcaSS_mean
    return ypred


def _fixed_get_pkcbLin(self, z=None, k=None):
    """CSSTemu ``PkcbLin_gp.get_pkcbLin``, with the same one-line fix.

    The HMF path reaches this: ``get_dndlnM`` -> ``get_dndlnM_Castro23`` ->
    ``_get_dlns_dlnR`` -> ``get_sigma_cb_z(type='Emulator')`` -> here.  So
    patching the mass function alone is not enough.

    ``self.__GPR`` and friends are name-mangled to ``_PkcbLin_gp__GPR``; this
    reaches them by their mangled names, which is why the replacement cannot
    simply be a subclass.
    """
    from CEmulator.emulator.PkLin import check_z, checkdata, RectBivariateSpline

    z = check_z(self.zlists, z)
    k = checkdata(self.klist, k, dname="wavenumber")
    if self.NormBeforeGP:
        Normcosmo = self.paramSS.transform(self.ncosmo)
    else:
        Normcosmo = np.copy(self.ncosmo)
    gpr = getattr(self, "_PkcbLin_gp__GPR")
    pkpred = np.zeros((self.nvec))
    for ivec in range(self.nvec):
        # The only change.
        pkpred[ivec] = np.ravel(gpr[ivec].predict(Normcosmo))[0]
    if self.NormBeforeGP:
        pkpred = self.pkcoeffSS.inverse_transform(pkpred.reshape(1, -1))[0]
    pkpred = 10 ** ((pkpred @ getattr(self, "_PkcbLin_gp__PCA_components"))
                    + getattr(self, "_PkcbLin_gp__PCA_mean"))
    pkpred = pkpred.reshape(len(self.zlists), len(self.klist))[::-1, :]
    spline = RectBivariateSpline(self.zlists[::-1], self.klist, pkpred, kx=3, ky=1)
    return spline(z, k)


def _fixed_get_pknn_cbLin(self, z=None, k=None):
    """CSSTemu ``Pknn_cbLin_gp.get_pknn_cbLin``, with the same one-line fix.

    A third method with the same defect, and it was found later than the other
    two because nothing in ``ggah_mod`` reaches it: the mass function goes
    through ``get_pkcbLin``, while this one is on the ``get_pklin`` /
    ``get_sigma8`` path.  ``emu_hmf`` reaches it, which is how a shim that
    claimed to make CSSTemu work under numpy 2 turned out to make two thirds
    of it work.

    Name-mangled to ``_Pknn_cbLin_gp__GPR``, as its neighbour is.
    """
    from CEmulator.emulator.PkLin import check_z, checkdata, RectBivariateSpline

    z = check_z(self.zlists, z)
    k = checkdata(self.klist, k, dname="wavenumber")
    if self.NormBeforeGP:
        Normcosmo = self.paramSS.transform(self.ncosmo)
    else:
        Normcosmo = np.copy(self.ncosmo)
    gpr = getattr(self, "_Pknn_cbLin_gp__GPR")
    pkpred = np.zeros((self.nvec))
    for ivec in range(self.nvec):
        # The only change.
        pkpred[ivec] = np.ravel(gpr[ivec].predict(Normcosmo))[0]
    if self.NormBeforeGP:
        pkpred = self.pkcoeffSS.inverse_transform(pkpred.reshape(1, -1))[0]
    pkpred = 10 ** ((pkpred @ getattr(self, "_Pknn_cbLin_gp__PCA_components"))
                    + getattr(self, "_Pknn_cbLin_gp__PCA_mean"))
    pkpred = pkpred.reshape(len(self.zlists), len(self.klist))[::-1, :]
    spline = RectBivariateSpline(self.zlists[::-1], self.klist, pkpred, kx=3, ky=1)
    return spline(z, k)


# ---- CEmulator.cosmology: three wrappers with the same defect --------------
#
# Each does ``out[iz] = <scalar> / self.get_Ez(z[iz]).reshape(-1)**2``.  get_Ez
# calls ``np.atleast_1d`` on its argument, so a scalar redshift comes back as a
# size-1 array, and the assignment into a scalar slot is the numpy-2 error
# again.  get_Ez itself is fine.
#
# The replacements are vectorised rather than patched line-by-line: the loop was
# only there to index a scalar out of an array, so removing it fixes the bug and
# is faster, and there is less of someone else's code copied into this file.

def _fixed_get_Omegam(self, z):
    z = np.atleast_1d(z)
    return self.Omegam * (1 + z) ** 3 / np.ravel(self.get_Ez(z)) ** 2


def _fixed_get_OmegaM(self, z):
    z = np.atleast_1d(z)
    return self.OmegaM * (1 + z) ** 3 / np.ravel(self.get_Ez(z)) ** 2


def _fixed_get_OmegaL(self, z):
    z = np.atleast_1d(z)
    de = np.exp(3 * ((1 / (1 + z) - 1) * self.wa
                     - (1 + self.w0 + self.wa) * np.log(1 / (1 + z))))
    return self.OmegaL * de / np.ravel(self.get_Ez(z)) ** 2


def ensure_cemulator_works(emulator) -> bool:
    """Probe the numpy-2 failure and patch it if present.

    Returns whether a patch was applied.  Raises anything that is *not* the
    known failure, because an unexpected error must not be silently patched
    over.
    """
    from CEmulator.cosmology import Cosmology as _CosmoCE
    from CEmulator.emulator.HMF import HMFbase_gp
    from CEmulator.emulator.PkLin import PkcbLin_gp, Pknn_cbLin_gp

    if getattr(HMFbase_gp, _FLAG, False):
        return True

    patched = bool(_restore_numpy_aliases())

    # The three cosmology wrappers, probed together: they share one defect, so
    # if one fails they all do.
    try:
        _CosmoCE.get_Omegam(emulator.cosmology, 0.5)
    except (ValueError, AttributeError):
        for name, fn in (("get_Omegam", _fixed_get_Omegam),
                         ("get_OmegaM", _fixed_get_OmegaM),
                         ("get_OmegaL", _fixed_get_OmegaL)):
            setattr(_CosmoCE, name, fn)
        patched = True
    for cls, method, replacement, probe in (
        (HMFbase_gp, "get_data", _fixed_get_data,
         lambda: emulator.HMFRockstarM200m.get_data()),
        (PkcbLin_gp, "get_pkcbLin", _fixed_get_pkcbLin,
         lambda: emulator.Pkcblin.get_pkcbLin(0.0, np.array([0.1]))),
        # The third.  Nothing in this package reaches it -- the mass function
        # goes through `get_pkcbLin` -- so it was missing until `emu_hmf`
        # called `get_sigma8` and found that a shim claiming to make CSSTemu
        # work under numpy 2 made two thirds of it work.
        (Pknn_cbLin_gp, "get_pknn_cbLin", _fixed_get_pknn_cbLin,
         lambda: emulator.Pknn_cblin.get_pknn_cbLin(0.0, np.array([0.1]))),
    ):
        try:
            probe()
            continue                       # upstream is fine; leave it alone
        except ValueError as exc:
            if "setting an array element with a sequence" not in str(exc):
                raise
        setattr(cls, method, replacement)
        patched = True

    setattr(HMFbase_gp, _FLAG, True)
    return patched
