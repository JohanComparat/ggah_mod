"""Write a tutorial notebook from a list of ("md" | "code", source) cells."""
import os

import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tutorials")

SETUP = r'''
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

from importlib.metadata import version

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import _helpers as T
T.paper_style()
FAST = T.fast_mode()
for name in ("ggah_mod", "emu_pk", "jax", "numpy"):
    print(f"{name:9s} {version(name)}")
print("FAST =", FAST)
'''


def write(name, cells):
    """Write ``tutorials/<name>.ipynb`` (sources only, no outputs)."""
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}}
    book.cells = [nbf.v4.new_markdown_cell(s.strip("\n")) if kind == "md"
                  else nbf.v4.new_code_cell(s.strip("\n")) for kind, s in cells]
    nbf.write(book, os.path.join(OUT, f"{name}.ipynb"))
    print("written", name)
