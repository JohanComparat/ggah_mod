# Contributing to ggah_mod

## The development install

> **Maintainer setup.** On the development laptop, use the shared `dev` environment
> defined in `~/software/dev_env` (`conda activate dev`); this package is already
> installed there in editable mode. Do not create a separate environment for it:
> add missing dependencies to `~/software/dev_env` and rebuild.

```bash
git clone https://github.com/JohanComparat/ggah_mod.git
cd ggah_mod
pip install -e ".[dev,reference]"
```

`[dev]` is the test suite; `pytest-xdist` is part of it because `pyproject.toml`
passes `-n auto`. `[reference]` compiles CLASS, which the `ACCURATE` flavour and a
few tests need; without it those tests skip by name.

## Checks

| target | runs |
|---|---|
| `make check` | all four below |
| `make test` | the suite, `pytest -q` |
| `make plan` | `tests/test_plan_is_true.py`: the README's and the maintainers' plan's statements that are facts about the code |
| `make contract` | `tests/test_downstream_contract.py`: what the maintainers' downstream repositories import from here |
| `make inbox` | `tests/test_cross_repo.py`: the maintainers' cross-repository notes resolve against the code |

The `Makefile` uses the maintainers' environment when it exists and `python`
otherwise; set `PY=/path/to/python` to choose. `pytest -p no:xdist` gives
readable output when a failure needs it, and `pytest -m "not slow"` skips the
tests that build a Boltzmann spectrum or execute the documentation.

**Coverage.** The `tests` workflow measures branch coverage of the core install
(no CLASS, `-m "not slow"`) and fails below `fail_under` in `pyproject.toml`; raise
the floor when coverage rises. The `full suite` workflow runs everything, the slow
tests and CLASS included, weekly and on demand.

The maintainers' development records (the plan of record and the
cross-repository notes, which docstrings and tests cite as `PLAN.md` items and
`CROSS_REPO.md` rows) and the downstream repositories are private. The tests that
audit them read them from `$GGAH_DEV_REPO`, `$GGAH_CAL_REPO` and
`$GGAH_BENCH_REPO`, and skip by name where they are absent.

## The documentation

The documentation is built with Sphinx and published on Read the Docs from
`docs/`. Its pages are MyST Markdown; one notebook per layer lives in
`docs/notebooks/`.

- **Notebooks are committed with their outputs.** Read the Docs renders them
  without executing them, because it cannot compile CLASS. `make docs` executes
  them first, in `$(PY)`, and then builds with `-W`; a notebook that no longer
  runs fails `pytest -m slow tests/test_docs.py`.
- **Tables are read off the code.** `docs/_ext/ggah_tables.py` writes the flavour
  table, the fiducial cosmology, the power-spectrum backends and the three
  layer-2 registries at build time. A registry entry without a citation in
  `docs/_ext/registry_citations.py` stops the build.
- **References come from the technical paper.** `docs/references.bib` is written
  by `docs/tools/sync_bib.py` from the paper's `references.bib` and is never
  edited by hand; `make docs-bib` fails if the two differ. Cite with
  `` {cite:t}`key` `` or `` {cite:p}`key` ``, using the paper's key.
- **The style is the paper's.** "We" for what was done or measured, "this
  package" for the code, the present tense for what it does; short paragraphs;
  no measured number typed into a page that a notebook or a generated table can
  produce instead.

```bash
pip install -e ".[docs]"
make docs            # execute the notebooks, then sphinx-build -W
make docs-bib        # docs/references.bib against the paper's bibliography
```

The built site is `docs/_build/html/index.html`.

## Releasing

1. Set the version in `pyproject.toml`, `ggah_mod/__init__.py` and
   `CITATION.cff` (with `date-released`), and add its entry to `CHANGELOG.md`.
2. Re-execute the notebooks (`make docs-notebooks`), so their printed versions
   are the release's, and run `make check` and `make docs`.
3. Tag `vX.Y.Z` on `main` and push the tag. `.github/workflows/release.yml`
   checks that the tag is the package's version, builds and checks the sdist and
   wheel, smoke-tests the wheel, publishes to PyPI by trusted publishing and
   creates the GitHub release.
