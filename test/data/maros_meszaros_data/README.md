# Maros Meszaros problems data files in .mat format

These are the converted Maros Meszaros problems to .mat files.
The problems have the form
```
minimize        0.5 x' P x + q' x + r

subject to      l <= A x <= u
```

where `x in R^n` is the optimization variable. The objective function is defined by a positive semidefinite matrix `P in S^n_+`, a vector `q in R^n` and a scalar `r in R`. The linear constraints are defined by matrix `A in R^{m x n}` and vectors `l in R^m U {-inf}^m`, `u in R^m U {+inf}^m`.



## Converting the problems

To generate the .mat files you need the following steps

1. Install [CUTEst](https://github.com/optimizers/cutest-mirror) with its Matlab interface. Make sure `cutest2matlab` works together with the interface commands.

2. From Matlab, run

    ```
    cd sif/
    extract_sif.m
    ```

The .mat files should appear in the `maros_meszaros_data/` folder. This repository already includes the converted files so that the scripts are easier to download and execute.


