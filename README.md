# LatticeABC (labc)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/labc/badge/?version=latest)](https://labc.readthedocs.io/)
[![Tests](https://github.com/ale-barone/labc/actions/workflows/tests.yml/badge.svg)](https://github.com/ale-barone/labc/actions/workflows/tests.yml)


A Python package for Lattice QCD data analysis, designed to be compatible
with standard NumPy functions.

The package is designed for reproducible analysis workflows in lattice field
theory, where observables are estimated from ensembles of gauge field
configurations, and uncertainties need to be propagated consistently through
non-linear transformations, fits, and derived quantities.

## Features

- **Statistical resampling** — jackknife and bootstrap with optional
  rebinning, accessible through a unified `StatsType` interface.
- **Error-aware containers** — `DataStats` and `DataErr` propagate errors
  automatically through basic arithmetic and arbitrary NumPy functions.
- **NumPy compatibility** — standard functions (`np.exp`, `np.log`, …) work directly on `DataStats` via the
  `__array_ufunc__` and `__array_function__` protocols.
- **Correlated fits** — least-squares and χ² minimisation with full
  covariance matrices, jackknife/bootstrap error estimation on fit parameters.
- **Plotting** — thin wrappers around Matplotlib to parse
  `DataStats` objects.


## Installation

At this stage, `labc` is intended to be used directly from the source
repository. Clone the repository with:

```bash
git clone https://github.com/ale-barone/labc.git
```

The package can then either be used directly from the source tree, or installed
in editable mode with:

```bash
pip install -e .
```

The editable installation is recommended (and encouraged) for development.

## Quick start

```python
import numpy as np
from labc import stats
from labc import data as dt

# resampling strategy
st = stats.StatsType.Jack(num_config=100)

# raw data: shape (num_config, T)
raw = np.random.normal(size=(100, 64))

# compute mean, error and jackknife bins
mean, err, bins = st.generate_stats(raw)

# store in a DataStats object
corr = dt.DataStats(mean, bins, st)

# error propagation is automatic, slicing act as in numpy
# e.g. effective mass
meff = -np.log(corr[1:]/corr[:-1])
```

## Documentation

Documentation, including an API reference, is available at
[labc.readthedocs.io](https://labc.readthedocs.io).


## Development status

`labc` is under active development and is used in lattice QCD analysis workflows.
The package is primarily developed for research use, with an emphasis on
transparent statistical analysis, reproducibility, and interoperability with the
scientific Python ecosystem.


## Requirements

`labc` requires Python ≥ 3.11.
Runtime dependencies and optional development dependencies are managed through
[`pyproject.toml`](pyproject.toml) and installed automatically with `pip`.

## License

This project is distributed under the MIT licence.