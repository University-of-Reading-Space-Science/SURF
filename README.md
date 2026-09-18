[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4889326.svg)](https://doi.org/10.5281/zenodo.4889326)
# HUXt - a computationally efficient solar wind model


## Introduction

This repository provides an implementation of the SURF (Space-weather Utilities for Research and Forecasting) modelling framework.

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

SURF disables Numba's on-disk compilation cache by default, which is useful
while developing. To enable it for production runs, change the following line
near the top of `surf/surf.py`:

```python
NUMBA_CACHE = True
```

Set it back to `False` for development. The setting is read when SURF is first
imported, so restart Python after changing it.



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
For testing with a development installation, a small test suite is included in ['test_surf.py'](tests/test_surf.py), which compares a local version of HUXt against a simple analytical solution and some reference simulation data included in this repository. The test suite uses `pytest`, which is included in the `surf` environment. Using the anaconda prompt from the root directory of `surf`, these tests can be performed by calling pytest from within the HUXt root directory:
```
pytest
```
The four tests should take around 1 minute to complete on a modest laptop. These tests are not an exhaustive test of all of the features in `surf`, but they do cover the core functionality and expected common use cases. They are mostly useful for anyone developing SURF, to check it is still working and how it differs from the latest available version. If you have installed SURF via pip or conda-forge, then the tests are completed as part of the conda-build process and not available to run locally.

## Usage

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
