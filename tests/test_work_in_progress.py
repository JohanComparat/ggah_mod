"""Layers 3 and beyond announce that they are work in progress; layers 1 and 2 do not.

Each check runs in a fresh interpreter: the notice is emitted once per process,
at the first import of a work-in-progress subpackage, and in this process those
imports happened long before any test ran.
"""
import subprocess
import sys

import pytest

WORK_IN_PROGRESS = ("sectors", "spectra", "observables", "covariance", "interfaces")


def _notices(code: str) -> list[str]:
    """The ``WorkInProgressWarning`` messages ``code`` raises, one per line."""
    probe = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as seen:\n"
        "    warnings.simplefilter('always')\n"
        + "".join(f"    {line}\n" for line in code.splitlines()) +
        "import ggah_mod\n"
        "for w in seen:\n"
        "    if issubclass(w.category, ggah_mod.WorkInProgressWarning):\n"
        "        print(w.message)\n")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_the_documented_layers_are_silent():
    assert _notices("import ggah_mod, ggah_mod.cosmology, ggah_mod.halos\n"
                    "import ggah_mod.halos.field, ggah_mod.halos.beyond_linear_bias") == []


@pytest.mark.parametrize("pkg", WORK_IN_PROGRESS)
def test_each_later_layer_warns_once_and_names_itself(pkg):
    notices = _notices(f"import ggah_mod.{pkg}")
    assert len(notices) == 1, notices
    assert notices[0].startswith(f"ggah_mod.{pkg} is work in progress")


def test_the_notice_can_be_silenced_on_its_own():
    assert _notices(
        "import ggah_mod\n"
        "warnings.filterwarnings('ignore', category=ggah_mod.WorkInProgressWarning)\n"
        "import ggah_mod.observables") == []
