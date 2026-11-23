# Paper I tutorials

These notebooks reproduce each section and each figure of Paper I, the cosmology and halo layers of `ggah_mod` and the cosmological constraints they give.
They are written for Master and PhD students.
Each notebook follows one section: the equations, the calls that compute them step by step, the figure of the paper, and exercises.

## Running them

```
pip install "ggah_mod[tutorials]"
cd tutorials
jupyter lab
```

Start with `00_getting_started.ipynb`.
The notebooks import `_helpers.py` from this directory.
`GGAH_TUTORIALS_FAST=0` turns off the reduced grids and reproduces the figures of the paper exactly; the default runs each notebook in a few minutes on a laptop CPU.
From a clone of the repository, `make tutorials` executes them all in place (`make tutorials FAST=0` on the paper's grids).

| Notebook | Paper I | Data bundle |
|---|---|---|
| `00_getting_started` | setup, flavours | no |
| `01_cosmology_background` | Secs. 2.1–2.2 | no |
| `02_power_spectrum` | Sec. 2.3 | no |
| `03_amplitude_growth` | Secs. 2.4–2.5 | no |
| `04_halo_variance` | Sec. 3.1 | no |
| `05_halo_abundance_bias` | Secs. 3.2–3.3 | no |
| `06_concentration_profiles` | Secs. 3.4–3.7 | no |
| `07_distance_constraints` | Sec. 4.1 | DESI DR2 posteriors |
| `08_full_shape` | Sec. 4.2 | DESI DR1 posteriors |
| `09_halo_counts` | Sec. 4.3 | no |
| `10_halo_requirements` | Sec. 4.4 | requirement scans |
| `11_joint_emulator_goal` | Secs. 4.5–4.6 | suite errors |
| `A_emulator_benchmark` | App. A | benchmark runs |
| `B_code_comparisons` | App. B | `accuracy.json` |

## The data bundle

The files the notebooks read are listed with their SHA-256 in `data_registry.txt`.
They are downloaded once from Zenodo ([doi:10.5281/zenodo.24469507](https://doi.org/10.5281/zenodo.24469507)) into `~/.cache/ggah_mod/tutorials`; `GGAH_TUTORIALS_DATA` points to a local copy instead.
`tools/build_tutorials_bundle.py` builds the bundle and rewrites the registry.
