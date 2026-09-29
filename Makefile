# `make check` is the suite plus the audits this repository owns; see
# CONTRIBUTING.md.  The maintainers' environment when it exists, else `python`.
MAINTAINER_PY := $(HOME)/mamba/envs/ggah_disco/bin/python
PY ?= $(if $(wildcard $(MAINTAINER_PY)),$(MAINTAINER_PY),python)
# The documentation build needs the `[docs]` extra; the notebooks are executed
# with $(PY), which has CLASS, so their ACCURATE cells run.
DOCS_PY ?= $(PY)

.PHONY: check test plan contract inbox docs docs-notebooks docs-html docs-bib

check: test plan contract inbox
	@echo "--- ggah_mod: check complete ---"

test:
	@echo "--- suite ---"
	@$(PY) -m pytest -q

plan:
	@echo "--- PLAN.md is true ---"
	@$(PY) -m pytest tests/test_plan_is_true.py -q

contract:
	@echo "--- the downstream contract, both siblings ---"
	@$(PY) -m pytest tests/test_downstream_contract.py -q

inbox:
	@echo "--- CROSS_REPO.md rows resolve ---"
	@$(PY) -m pytest tests/test_cross_repo.py -q

docs: docs-notebooks docs-html

docs-notebooks:
	@echo "--- execute the notebooks in place ---"
	@$(PY) docs/tools/run_notebooks.py

docs-html:
	@echo "--- sphinx, warnings are errors ---"
	@$(DOCS_PY) -m sphinx -W --keep-going -b html docs docs/_build/html

docs-bib:
	@echo "--- docs/references.bib is the paper's ---"
	@$(PY) docs/tools/sync_bib.py --check
