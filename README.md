# myEvaPy

Using Numba.njit to speed up extreme value analysis (EVA)

Generally, the lmoments3 is used to carry out extreme value analysis. 
It works perfect for single grid.

But when it is applied to xarray + dask with million grids, it takes time.
Therefore, rewrite some functions with numba.njit to speed up the analysis

## supported distributions
- gamma
- gev
- gumbel
- Pearson III
- Weibull distribution

## extra functions
- Add bootstrapping for confident intervals

## external dependency - rvlib
conda config --add channels conda-forge
conda install rvlib

The package due to change, and may update!
