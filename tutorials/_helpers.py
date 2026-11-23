"""Shared pieces of the Paper I tutorials: the data files, the figure style.

The notebooks import this file from their own directory (``import _helpers``).
It is not part of the installed package.

:func:`data_path` returns the local path of a file of the tutorial data
bundle.  The bundle is looked for, in order, in the directory named by the
environment variable ``GGAH_TUTORIALS_DATA``, then in the cache
(``~/.cache/ggah_mod/tutorials``), and is otherwise downloaded once from Zenodo
with :mod:`pooch`.  Each file is checked against ``data_registry.txt``, the
SHA-256 of every file in the bundle, so a notebook reads exactly the file the
paper was made from.
"""
from __future__ import annotations

import hashlib
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "ggah_mod", "tutorials")
BUNDLE = "ggah_mod_tutorials_data"

#: The Zenodo archive of the bundle, doi:10.5281/zenodo.24469507.
ZENODO_URL = ("https://zenodo.org/records/24469507/files/"
              "ggah_mod_tutorials_data-4a08dac.tar.gz")
ZENODO_SHA256 = "b30c5ab63169f834f6b91171600a6c67893b439fa9a38ba039431d50026ff548"

#: The figure style of the paper.
PAPER_RC = {
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
    "axes.titlesize": 8, "axes.grid": True, "grid.alpha": 0.25,
    "lines.linewidth": 1.6,
}

#: LaTeX labels of the fields of :class:`ggah_mod.cosmology.Cosmology`.
TEX = {"Omega_m": r"$\Omega_{\rm m}$", "Omega_b": r"$\Omega_{\rm b}$",
       "h": r"$h$", "sum_mnu": r"$\sum m_\nu$", "T_cmb": r"$T_{\rm CMB}$",
       "n_s": r"$n_{\rm s}$", "ln10A_s": r"$\ln(10^{10}A_{\rm s})$",
       "w0": r"$w_0$", "wa": r"$w_a$", "Omega_k": r"$\Omega_k$",
       "sigma8": r"$\sigma_8$"}

#: Parameters given a logarithmic derivative (positive scales) and an
#: absolute one (locations, or already logarithms), as in the paper.
DLOG = ("Omega_m", "Omega_b", "h", "sum_mnu", "T_cmb")
DABS = ("n_s", "ln10A_s", "w0", "wa", "Omega_k")
DPARAMS = DLOG + DABS

#: Colour and line style of each parameter in the gradient figures.
DSTYLE = {
    "Omega_m": ("C0", "-"), "Omega_b": ("C0", (0, (1, 1.4))),
    "h": ("C1", "-"), "sum_mnu": ("C2", "-"),
    "T_cmb": ("C1", (0, (1, 1.4))),
    "n_s": ("C4", (0, (1, 1.4))), "ln10A_s": ("C5", (0, (1, 1.4))),
    "w0": ("C3", "-"), "wa": ("C3", (0, (5, 1.6))),
    "Omega_k": ("C6", "-"),
}


def paper_style():
    """Apply the figure style of the paper to matplotlib."""
    plt.rcParams.update(PAPER_RC)


def save_figure(fig, name, **kw):
    """Save ``fig`` as ``figures/<name>.pdf`` next to the notebooks, as the paper saves it."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.pdf"), **kw)


def fast_mode(default=True):
    """Return whether to run on reduced grids: ``GGAH_TUTORIALS_FAST``, else ``default``."""
    value = os.environ.get("GGAH_TUTORIALS_FAST")
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "")


def _registry():
    out = {}
    with open(os.path.join(HERE, "data_registry.txt")) as f:
        for line in f:
            if line.strip():
                sha, name = line.split(maxsplit=1)
                out[name.strip()] = sha
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download():
    """Download and unpack the bundle into the cache; return its directory."""
    if ZENODO_URL is None:
        raise RuntimeError(
            "the tutorial data is not on Zenodo yet: set GGAH_TUTORIALS_DATA to a "
            "local copy of the bundle")
    import pooch
    pooch.retrieve(ZENODO_URL, known_hash=f"sha256:{ZENODO_SHA256}", path=CACHE,
                   processor=pooch.Untar(extract_dir="."))
    return os.path.join(CACHE, BUNDLE)


def data_path(name):
    """Return the local path of the bundle file ``name``, checked against its SHA-256.

    ``name`` is the path inside the bundle, e.g. ``"desi/desi_dr2_contours.json"``.
    """
    registry = _registry()
    if name not in registry:
        raise KeyError(f"{name!r} is not in data_registry.txt")
    roots = [os.environ.get("GGAH_TUTORIALS_DATA"), os.path.join(CACHE, BUNDLE)]
    root = next((r for r in roots if r and os.path.isfile(os.path.join(r, name))), None)
    if root is None:
        root = _download()
    path = os.path.join(root, name)
    if _sha256(path) != registry[name]:
        raise RuntimeError(f"{path} does not match data_registry.txt")
    return path


class DesiPosteriors:
    """DESI's sampled posteriors, reduced to 68 per cent contours and marginals.

    ``name`` is a bundle file: ``"desi/desi_dr2_contours.json"`` (DR2 BAO,
    chains ``"bao"`` and ``"bbn"``) or ``"desi/desi_dr1_fs_contours.json"``
    (DR1 full shape).  Parameters are named as in
    :class:`ggah_mod.cosmology.Cosmology`.
    """

    def __init__(self, name):
        import json
        with open(data_path(name)) as f:
            self.data = json.load(f)

    def chains(self):
        """The chains in the file."""
        return list(self.data["chains"])

    def parameters(self, chain):
        """The parameters ``chain`` constrains, in figure order."""
        return list(self.data["chains"][chain]["parameters"])

    def contours(self, chain, px, py):
        """The 68 per cent region in the ``(px, py)`` plane, as closed ``(N, 2)`` polygons."""
        import numpy as np
        c = self.data["chains"][chain]
        if px not in c["parameters"] or py not in c["parameters"]:
            return []
        key, swap = f"{px}|{py}", False
        if key not in c["contours"]:
            key, swap = f"{py}|{px}", True
        polys = [np.column_stack([p["x"], p["y"]]) for p in c["contours"][key]]
        return [p[:, ::-1] for p in polys] if swap else polys

    def marginal(self, chain, p):
        """``(x, P)``: the 1-D posterior of ``p``, peak normalised to 1."""
        import numpy as np
        m = self.data["chains"][chain]["marginals"][p]
        return np.asarray(m["x"]), np.asarray(m["P"])

    def limits68(self, chain, p):
        """``(mean, sd, lower, upper, kind)`` of the 68 per cent marginal interval."""
        m = self.data["chains"][chain]["limits68"][p]
        return m["mean"], m["sd"], m["lower"], m["upper"], m["kind"]

    def provenance(self, chain=None):
        """The header fields of the file, or of one chain."""
        d = self.data if chain is None else self.data["chains"][chain]
        return {k: v for k, v in d.items() if k.startswith("_")}


def fisher_ellipse(ax, mean, cov, *, k=1.5165750888103102, **kw):
    """Draw the 68 per cent contour of a 2-D Gaussian (Delta chi^2 = 2.30) as a patch."""
    import numpy as np
    from matplotlib.patches import Polygon
    w, v = np.linalg.eigh(cov)
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    pts = v @ (k * np.sqrt(np.clip(w, 0.0, None))[:, None] * np.array([np.cos(t), np.sin(t)]))
    ax.add_patch(Polygon(np.column_stack([mean[0] + pts[0], mean[1] + pts[1]]), closed=True, **kw))
