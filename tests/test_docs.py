r"""The documentation says what the package has, and its code runs.

``README.md``, the pages under ``docs/`` and the notebooks are read by
people who will paste what they say.  ``PLAN.md`` Block F5 once removed
``docs/`` on the ground that a rendered copy of the docstrings goes stale and a
stale page is worse than none -- true of a page nothing checks.  These tests are
the checking:

* the README's quickstart runs, in a fresh interpreter, on the install the
  core CI job makes (no CLASS);
* the README names every subpackage and lists every extra with exactly the
  packages ``pyproject.toml`` installs for it;
* every citation resolves in ``docs/references.bib`` and every entry there is
  cited, and every registry entry the generated tables list has a source;
* the registry sizes the layer pages state in words are the registries' sizes;
* **slow**: every page's snippets run in order, and every notebook executes.

What they cannot check is prose; the numbers a page quotes come from the
generated tables and the executed notebooks, so that there is little prose
left that states one.
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
PAGES = ("overview.md", "layer1_cosmology.md", "layer2_halos.md",
         "layer3_galaxies.md")
NOTEBOOKS = sorted((DOCS / "notebooks").glob("*.ipynb"))

_PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
_WORDS = {"two": 2, "six": 6, "seven": 7, "thirteen": 13, "sixteen": 16,
          "eighteen": 18}


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    assert start in text and end in text, f"{start} / {end} missing"
    return text.split(start, 1)[1].split(end, 1)[0]


def _section(text: str, heading: str) -> str:
    """From ``heading`` to the next heading of the same level."""
    level = heading.split(" ", 1)[0]
    body = text.split(heading + "\n", 1)[1]
    return re.split(rf"^{level} ", body, maxsplit=1, flags=re.M)[0]


def _run(code: str, timeout: float = 1800.0) -> str:
    """In a fresh interpreter, as a reader would: x64 must be set before the
    first import, and an in-process run would inherit the suite's setting."""
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True,
                          capture_output=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# The README
# ==========================================================================
def test_the_readme_quickstart_runs():
    block = _between(_readme(), "<!-- quickstart-start -->", "<!-- quickstart-end -->")
    code = _PYTHON_BLOCK.search(block).group(1)
    _run(code)


def test_the_readme_layer_map_names_every_subpackage():
    table = _section(_readme(), "## The six layers")
    packages = sorted(p.parent.name for p in (ROOT / "ggah_mod").glob("*/__init__.py"))
    missing = [p for p in packages if f"`ggah_mod.{p}`" not in table]
    assert not missing, f"the README's layer map omits {missing}"


def test_the_readme_extras_are_pyprojects():
    """The install table lists each extra with the packages it installs."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = extras["project"]["optional-dependencies"]
    want = {name: sorted(re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0] for req in reqs)
            for name, reqs in extras.items()}

    block = _between(_readme(), "<!-- install-start -->", "<!-- install-end -->")
    have = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"`[a-z]+`", cells[0]):
            have[cells[0].strip("`")] = sorted(re.findall(r"`([^`]+)`", cells[1]))
    assert have == want


# ==========================================================================
# The pages
# ==========================================================================
def test_the_environment_file_declares_nothing_pyproject_does():
    """``environment.yml`` installs ``ggah_mod`` through pip and restates none of
    its requirements, so a version floor lives in ``pyproject.toml`` alone.  It
    once carried ``emu-pk >=1.0.0,<2`` for hours after the floor had moved."""
    env = (ROOT / "environment.yml").read_text(encoding="utf-8")
    lines = [ln.split("#", 1)[0].strip() for ln in env.splitlines()]
    entries = [ln[2:].strip() for ln in lines if ln.startswith("- ")]
    assert "ggah_mod" in entries, "environment.yml no longer installs ggah_mod"

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = {re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0].replace("-", "_")
              for req in project["project"]["dependencies"] if re.search(r"[<>=]", req)}
    restated = [e for e in entries
                if re.split(r"[<>=!~ \[]", e, maxsplit=1)[0].replace("-", "_") in pinned]
    assert not restated, f"environment.yml restates pyproject's floors: {restated}"

    extras = set(project["project"]["optional-dependencies"])
    named = set(re.findall(r"ggah_mod\[([a-z,]+)\]", env)) | set(
        re.findall(r"`\[([a-z]+)\]`", env))
    named = {x for group in named for x in group.split(",")}
    assert named <= extras, f"environment.yml names extras pyproject lacks: {named - extras}"


@pytest.mark.parametrize("text_of, phrase, registry", [
    ("page", "multiplicity functions", "MULTIPLICITY"),
    ("page", "bias fits", "BIAS"),
    ("page", "concentration relations", "CONCENTRATION"),
    ("module", "multiplicity functions", "MULTIPLICITY"),
])
def test_the_layer2_counts_are_the_registries(text_of, phrase, registry):
    import ggah_mod.halos as H
    from ggah_mod.halos.mass_function import COSMOLOGY_DEPENDENT_MULTIPLICITY

    text = ((DOCS / "layer2_halos.md").read_text(encoding="utf-8")
            if text_of == "page" else H.__doc__)
    pattern = r"(\w+)\s+" + r"\s+".join(phrase.split())
    words = [w.lower() for w in re.findall(pattern, text)]
    assert words, (text_of, phrase)
    size = len(getattr(H, registry))
    assert {_WORDS[w] for w in words} == {size}, (text_of, phrase, words, size)
    if registry == "MULTIPLICITY":
        published = [w.lower() for w in re.findall(r"\((\w+)\s+published\s+fits", text)]
        assert {_WORDS[w] for w in published} == {
            size - len(COSMOLOGY_DEPENDENT_MULTIPLICITY)}


@pytest.mark.parametrize("phrase, module, registry", [
    ("occupation models", "occupation", "OCCUPATION"),
    ("conditional luminosity functions", "clf", "CLF"),
    ("stellar-mass relations", "sham", "SHMR"),
])
def test_the_layer3_counts_are_the_registries(phrase, module, registry):
    """Every "<number> <phrase>" on the galaxy page names the registry's size."""
    import importlib

    text = (DOCS / "layer3_galaxies.md").read_text(encoding="utf-8")
    pattern = r"(\w+)\s+" + r"\s+".join(re.escape(w) for w in phrase.split())
    words = [w.lower() for w in re.findall(pattern, text)]
    assert words, phrase
    size = len(getattr(importlib.import_module(f"ggah_mod.sectors.{module}"), registry))
    assert {_WORDS[w] for w in words} == {size}, (phrase, words, size)


def test_every_citation_is_in_the_bibliography_and_every_entry_is_cited():
    sync = _load(DOCS / "tools" / "sync_bib.py", "_sync_bib_for_docs")
    cited = sync.cited()
    have = set(sync.entries((DOCS / "references.bib").read_text(encoding="utf-8")))
    assert not cited - have, f"cited but not in docs/references.bib: {sorted(cited - have)}"
    assert not have - cited, f"in docs/references.bib but never cited: {sorted(have - cited)}"


def test_every_registry_entry_has_a_source():
    """What ``docs/_ext/ggah_tables.py`` enforces at build time, here as well,
    so that a new fit fails the suite and not only the documentation job."""
    import ggah_mod.halos as H
    from ggah_mod.sectors import clf, occupation, sham

    cite = _load(DOCS / "_ext" / "registry_citations.py", "_registry_citations")
    for name in ("MULTIPLICITY", "BIAS", "CONCENTRATION"):
        assert set(getattr(cite, name)) == set(getattr(H, name)), name
    for name, module in (("OCCUPATION", occupation), ("CLF", clf), ("SHMR", sham)):
        assert set(getattr(cite, name)) == set(getattr(module, name)), name


# ==========================================================================
# Execution
# ==========================================================================
@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_every_snippet_on_a_page_runs(page):
    """In order, in one interpreter per page: a page's snippets build on each
    other, as a reader running them would."""
    blocks = _PYTHON_BLOCK.findall((DOCS / page).read_text(encoding="utf-8"))
    assert blocks or page == "overview.md"
    _run("\n".join(blocks))


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_every_notebook_executes(path):
    """Executed on a copy: the committed outputs are rewritten by
    ``make docs-notebooks``, deliberately, and never by the suite."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")

    nb = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    client = nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                                     resources={"metadata": {"path": str(path.parent)}})
    client.execute()


def test_the_notebooks_are_committed_executed():
    """Read the Docs renders what is stored; an unexecuted notebook would
    publish as code with no output."""
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
        errors = [i for i, c in enumerate(code)
                  for o in c.get("outputs", []) if o.get("output_type") == "error"]
        assert not unrun and not errors, (path.name, unrun, errors)
