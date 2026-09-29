"""Sphinx configuration for the ggah_mod documentation.

Read the Docs installs ``pip install .[docs]``: the package and its two emulators
from PyPI, no Boltzmann solver, no Jupyter kernel.  That is enough because
nothing here is computed by the build except the tables ``_ext/ggah_tables.py``
reads off the registries -- the notebooks are rendered from their committed
outputs (``nb_execution_mode = "off"``), and ``make docs`` is what executes them.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext"))

import ggah_mod                                                   # noqa: E402

project = "ggah_mod"
author = "Johan Comparat"
copyright = "2026, Johan Comparat"
# From the package, not from `importlib.metadata`: an editable install keeps the
# metadata of whatever version it was first installed at.
release = ggah_mod.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "ggah_tables",
]

master_doc = "index"
exclude_patterns = ["_build", "_generated", "jupyter_execute", "tools",
                    "**.ipynb_checkpoints"]

# --- MyST and notebooks ------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
myst_dmath_double_inline = True
nb_execution_mode = "off"
nb_merge_streams = True

# --- API ---------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
# Only `interfaces/cobaya.py` imports an optional package at module scope; the
# rest are imported inside the functions that need them.  Listed anyway so that
# an import moving to module scope cannot break the build on Read the Docs.
autodoc_mock_imports = ["classy", "camb", "CEmulator", "soxs", "healpy", "cobaya"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# `Attributes` sections as fields of the class, not as attribute entries: a
# dataclass field is already documented by autodoc from its annotation, and two
# entries for one name is a duplicate-object warning under `-W`.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# --- Bibliography ------------------------------------------------------------
# `references.bib` is written by `tools/sync_bib.py` from the technical paper's
# bibliography; see that script.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year_round"
bibtex_default_style = "authoryear"


def _round_brackets():
    """"Hogg (1999)" and "(Hogg 1999)", as the paper cites, not "Hogg [1999]"."""
    import dataclasses

    from sphinxcontrib.bibtex.plugin import register_plugin
    from sphinxcontrib.bibtex.style.referencing import BracketStyle
    from sphinxcontrib.bibtex.style.referencing.author_year import (
        AuthorYearReferenceStyle)

    def round_():
        return dataclasses.field(
            default_factory=lambda: BracketStyle(left="(", right=")"))

    @dataclasses.dataclass
    class RoundAuthorYear(AuthorYearReferenceStyle):
        bracket_parenthetical: BracketStyle = round_()
        bracket_textual: BracketStyle = round_()
        bracket_author: BracketStyle = round_()
        bracket_label: BracketStyle = round_()
        bracket_year: BracketStyle = round_()

    register_plugin("sphinxcontrib.bibtex.style.referencing",
                    "author_year_round", RoundAuthorYear)


_round_brackets()


def _author_year_style():
    """The bibliography labelled and sorted the way the text cites it."""
    from pybtex.plugin import register_plugin
    from pybtex.style.formatting.unsrt import Style
    from pybtex.style.labels import BaseLabelStyle

    class AuthorYearLabels(BaseLabelStyle):
        def format_labels(self, sorted_entries):
            for entry in sorted_entries:
                people = entry.persons.get("author", [])
                last = [" ".join(p.last_names) for p in people]
                if len(last) > 2:
                    who = f"{last[0]} et al."
                elif len(last) == 2:
                    who = f"{last[0]} & {last[1]}"
                else:
                    who = last[0] if last else entry.key
                who = who.replace("{", "").replace("}", "")
                yield f"{who} {entry.fields.get('year', '')}"

    class AuthorYearStyle(Style):
        default_label_style = AuthorYearLabels
        default_sorting_style = "author_year_title"

    register_plugin("pybtex.style.formatting", "authoryear", AuthorYearStyle)


_author_year_style()

# --- Output ------------------------------------------------------------------
html_theme = "furo"
html_title = f"ggah_mod {release}"
html_theme_options = {
    "source_repository": "https://github.com/JohanComparat/ggah_mod/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# MathJax has no `\dd`, and the paper's shorthands are not primitives either.
# `-W` cannot see an undefined macro: MathJax runs in the reader's browser after
# the build has passed, and prints it in red.  So every macro a page or a
# docstring uses is defined here, with the paper's definition.
mathjax3_config = {
    "tex": {
        "macros": {
            "dd": r"\mathrm{d}",
            "rhom": r"\bar{\rho}_{\rm m}",
            "rhocb": r"\bar{\rho}_{\rm cb}",
            "Plin": r"P_{\rm lin}",
            "Pm": r"P_{\rm m}",
            "Pcb": r"P_{\rm cb}",
            "Ocb": r"\Omega_{\rm cb}",
            "Onum": r"\Omega_\nu^{\rm nr}",
            "Onur": r"\Omega_\nu^{\rm r}",
            "Onuur": r"\Omega_\nu^{\rm ur}",
            "Our": r"\Omega_{\rm ur}",
            "Ode": r"\Omega_{\rm DE}",
            "summnu": r"\textstyle\sum m_\nu",
            "lnA": r"\ln(10^{10}A_{\rm s})",
        },
    },
}
