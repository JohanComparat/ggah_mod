# Paper I tutorials

These notebooks reproduce each section and each figure of Paper I, the cosmology and halo layers of `ggah_mod` and the cosmological constraints they give.
They are written for Master and PhD students.
Each notebook follows one section: the equations, the calls that compute them step by step, the figure of the paper, and exercises.

## Running them

The notebooks are not part of the PyPI package: clone the repository, install the extra and start Jupyter in this directory.

```
git clone https://github.com/JohanComparat/ggah_mod.git
pip install "ggah_mod[tutorials]"
cd ggah_mod/tutorials
jupyter lab
```

Start with `00_getting_started.ipynb`.
The notebooks import `_helpers.py` from this directory.
`GGAH_TUTORIALS_FAST=0` turns off the reduced grids and reproduces the figures of the paper exactly; the default runs on reduced grids.
The table gives the wall time of each notebook in both modes, on a laptop (Intel Core Ultra 9 285HX, 24 threads, CPU only, ggah_mod 1.7): 295 s for all fourteen on the reduced grids and 443 s on the paper's.
The data bundle is downloaded once, in a few seconds, by the first notebook that reads it.
From the top of the clone, `make tutorials` executes them all in place (`make tutorials FAST=0` on the paper's grids).

| Notebook | Paper I | Data bundle | Reduced grids | Paper's grids |
|---|---|---|---|---|
| `00_getting_started` | setup, flavours | no | 5 s | 8 s |
| `01_cosmology_background` | Secs. 2.1–2.2 | no | 9 s | 12 s |
| `02_power_spectrum` | Sec. 2.3 | no | 8 s | 18 s |
| `03_amplitude_growth` | Secs. 2.4–2.5 | no | 16 s | 43 s |
| `04_halo_variance` | Sec. 3.1 | no | 7 s | 13 s |
| `05_halo_abundance_bias` | Secs. 3.2–3.3 | no | 14 s | 23 s |
| `06_concentration_profiles` | Secs. 3.4–3.7 | no | 25 s | 45 s |
| `07_distance_constraints` | Sec. 4.1 | DESI DR2 posteriors | 7 s | 10 s |
| `08_full_shape` | Sec. 4.2 | DESI DR1 posteriors | 30 s | 53 s |
| `09_halo_counts` | Sec. 4.3 | no | 91 s | 122 s |
| `10_halo_requirements` | Sec. 4.4 | requirement scans | 70 s | 82 s |
| `11_joint_emulator_goal` | Secs. 4.5–4.6 | suite errors | 6 s | 7 s |
| `A_emulator_benchmark` | App. A | benchmark runs | 4 s | 4 s |
| `B_code_comparisons` | App. B | `accuracy.json` | 3 s | 3 s |

## The data bundle

The files the notebooks read are listed with their SHA-256 in `data_registry.txt`.
They are downloaded once from Zenodo ([doi:10.5281/zenodo.24469507](https://doi.org/10.5281/zenodo.24469507)) into `~/.cache/ggah_mod/tutorials`; `GGAH_TUTORIALS_DATA` points to a local copy instead.
`tools/build_tutorials_bundle.py` builds the bundle and rewrites the registry.
