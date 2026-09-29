r"""Adapters that let other packages drive this one.

Each module here imports a third-party package at module level and is therefore
**not** imported by :mod:`ggah_mod`: you pay for an interface only by asking for
it.  That is why this subpackage re-exports nothing.
"""

from .._wip import warn_work_in_progress

warn_work_in_progress(__name__)

__all__: list[str] = []
