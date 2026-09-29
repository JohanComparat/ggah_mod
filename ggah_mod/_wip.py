"""The work-in-progress notice of layer 3 and beyond.

Layers 1 and 2 (``cosmology``, ``halos``) are verified and documented.  The
tracers, the spectra, the observables and the covariance (layers 3 to 6) and the
interfaces are released as code, tested, ahead of their verification and their
documentation.  Importing any of them warns once per process, with
:class:`WorkInProgressWarning`, which can be silenced on its own::

    import warnings
    import ggah_mod
    warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)
"""
import warnings

__all__ = ["WorkInProgressWarning", "warn_work_in_progress"]

_warned = False


class WorkInProgressWarning(UserWarning):
    """A subpackage whose interface and results may still change."""


def warn_work_in_progress(module: str) -> None:
    """Warn that ``module`` is work in progress, the first time any such
    subpackage is imported: one import of ``ggah_mod.observables`` loads the
    spectra and the tracers too, and one notice says it."""
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(
        f"{module} is work in progress. Layers 3 to 6 of ggah_mod (sectors, "
        f"spectra, observables, covariance) and its interfaces are released as "
        f"tested code ahead of their verification and documentation; their "
        f"interface and results may change. Layers 1 and 2 (cosmology, halos) "
        f"are documented at https://ggah-mod.readthedocs.io.",
        WorkInProgressWarning, stacklevel=2)
