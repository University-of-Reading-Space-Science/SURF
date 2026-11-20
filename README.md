[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4889326.svg)](https://doi.org/10.5281/zenodo.4889326)
# SURF - Space Weather Utilities for Research and Forecasting


## Introduction

This repository provides an implementation of the SURF modelling framework.

This includes the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3). This is a simple 1D reduced hydrodynamic model, which essentially solves Burgers equation using the upwind numerical scheme. For more details on the models background, refer to [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3).

## Installation
 `SURF` is written in Python 3.12.11 and has a range of dependencies, which are listed in the `environment.yml` file.

This simplest way to work with SURF is to use `conda`, and we recommend using an up-to-date version of [miniforge](https://conda-forge.org/download/). 
 
SURF can be installed via conda-forge. We recommend installing HUXt into a virtual environment and this can be done as:

```
>>> conda create --name surf surfs-up
>>> conda activate surf
```

## Development Installation
If you are developing features in SURF, it can be easier to work with an editable installation. To do this, it is easiest to clone this repository, and with the anaconda prompt in the HUXt root directory:

```
>>> conda env create -f environment.yml
>>> conda activate surf
>>> pip install --no-deps -e .
```

### Numba cache mode

SURF enables Numba's on-disk compilation cache by default, as in general it makes simulations 
faster. However, for model development work it can be useful to disable this. To disable it for 
development work, change line 2515 of `surf/surf.py`:

```python
numba_cache = True
```

Set it back to `True` for general use.

Installation also provides the `surf-open-examples` and
`surf-make-ephemeris` command-line tools.

### SURFs_UP web interface

SURFs_UP is included as the `surfs_up` package. Start its local web interface
with either command:

```powershell
surfs-up
```

Then open `http://127.0.0.1:5000`, if it does not open automatically. The local launcher runs the background job worker in the same process. For a WSGI deployment, run `surfs-up-worker` as a
separate persistent process.


### Testing
With a development installation, run the test suite from the repository root with `pytest` (included in the `surf` environment):

```
pytest
```

The suite takes around a minute on a modest laptop; the first run takes longer while Numba compiles and caches the solvers. [`tests/test_surf.py`](tests/test_surf.py) contains two kinds of test:

* Verification tests check behaviour any correct version of SURF must show: the HUXt steady state matches its analytic acceleration profile, the hydro solver conserves mass flux, parallel runs match serial runs, and saved runs reload unchanged.
* Regression tests rerun reference simulations covering the HUXt, hydro and MHD solvers (including CMEs, magnetic polarity, streaklines, time-dependent boundaries and chunked runs) and compare them with `tests/reference_data/surf_regression.npz`. They fail if SURF's numerical output changes.

If a change to the output is intended, for example a bug fix, regenerate the reference data with `python tests/generate_reference_data.py` and record the reason in the commit message.

## Usage

SURF also includes a native spherical ideal-MHD solver: select `solver='mhd'`
and provide physical fields through `mhd_boundary`, for example
`{'br': 10*u.nT, 'bt': -20*u.nT}`. It uses the existing simulation, tracer,
restart and HDF5 framework without an external MHD code. See the
[MHD guide and model review](docs/mhd.md) for examples, equations, units,
boundary constraints and verification.

Some examples of how to use HUXt can be found in [`surf_examples.ipynb`](huxt/notebooks/HUXt_examples.ipynb), which can be opened using the `surf-open-examples` command after installation.

`SURF` requires an inner boundary condition for longitudinal solar wind speed profile. This can either be prescribed by the user or derived from other sources. For convenience,  [`surf_inputs.py`](code/huxt_inputs.py) provides some functions for downloading and generating longitudinal solar wind speed profiles from the [HelioMAS](https://doi.org/10.1029/2000JA000121), Wang-Sheeley-Arge (WSA), and Potential Field Source Surface (PFSS) models, as well as from Coronal Tomography (CorTom). Examples of each is provided in the examples workbook. Routines for plotting and animating HUXt solutions can be found in  [`surf_analysis.py`](code/huxt_analysis.py). Again, examples are provided in the workbook.

## Contact
Please contact either [Mathew Owens](https://github.com/mathewjowens) or [Luke Barnard](https://github.com/lukebarnard). 

## Citations

If you use SURF in a publication or presentation, please cite the software using the Zenodo reference with DOI:[10.5281/zenodo.4889326](https://doi.org/10.5281/zenodo.4889326) 

To cite this project, including the scientific basis and functionality of SURF, please use: 

Owens and Barnard. (2026), *Space-weather Utilities for Research and Forecasting (SURF): A tool for investigating hydrodynamic aspects of solar wind and coronal mass ejection expansion and evolution*, Solar Physics [doi](https://www.researchsquare.com/article/rs-9496956/v1)

Barnard and Owens. (2022), *HUXt - An open source, computationally efficient reduced-physics solar wind model, written in Python*, Frontiers in Physics [10.3389/fphy.2022.1005621](https://doi.org/10.3389/fphy.2022.1005621)

Owens et al. (2020),  *A Computationally Efficient, Time-Dependent Model of the Solar Wind for Use as a Surrogate to Three-Dimensional Numerical Magnetohydrodynamic Simulations*,  Sol Phys, DOI:[10.1007/s11207-020-01605-3.svg](https://doi.org/10.1007/s11207-020-01605-3)
