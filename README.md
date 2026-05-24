# LatticeABC (labc)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/labc/badge/?version=latest)](https://labc.readthedocs.io/)
[![Tests](https://github.com/ale-barone/labc/actions/workflows/tests.yml/badge.svg)](https://github.com/ale-barone/labc/actions/workflows/tests.yml)


A Python package for Lattice QCD data analysis, designed to be compatible
with standard NumPy functions.

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

```bash
git clone https://github.com/ale-barone/LatticeABC.git labc
cd labc
pip install -e .
```

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

Full documentation (work in progress) with API reference is available at
[labc.readthedocs.io](https://labc.readthedocs.io).

## Requirements

Python ≥ 3.11.  All dependencies are listed in
[`pyproject.toml`](pyproject.toml) and installed automatically with `pip`.