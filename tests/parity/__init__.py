"""Parity suite: this package measured against the ``hod_mod`` reference.

Exists as a package so a parity file can share a basename with the unit-test
file for the same module -- ``tests/test_x.py`` and ``tests/parity/test_x.py``
are then distinct modules to pytest, and the parity file can go on mirroring the
name of what it compares.

``hod_mod`` is a test-only dependency; nothing under ``ggah_mod/`` imports it.
Each file here skips itself when it is not installed.
"""
