r"""Adapters that let other packages drive this one.

Each module here imports a third-party package at module level, so
:mod:`ggah_mod` does not import it and this subpackage re-exports nothing: an
interface costs an import only when it is asked for.
"""

from .._wip import warn_work_in_progress

warn_work_in_progress(__name__)

__all__: list[str] = []
