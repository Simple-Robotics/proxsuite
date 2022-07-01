# The PROXQP Solver

![License](https://img.shields.io/badge/License-BSD%202--Clause-green.svg)

The PROXQP solver is a numerical optimization package for solving problems in the form
```
minimize        0.5 x' H x + g' x

subject to           A x = b
                l <= C x <= u
```

where `x in R^n` is the optimization variable. The objective function is defined by a positive semidefinite matrix `H in S^n_+` and vector `g in R^n`. The linear constraints are defined by matrices `A in R^{n_eq x n}`, `C in R^{n_in x n}` and vectors `b`, `l` and `u` so that `b_i in R` for all `i in 1,...,n_eq` and `l_i in R U {-inf}` and `u_i in R U {+inf}` for all `i in 1,...,n_in`.

## Citing PROXQP

If you are using PROXQP for your work, we encourage you to [cite the related paper](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/).

## Numerical benchmarks

Numerical benchmarks against other solvers are available [here](https://github.com/Bambade/proxqp_benchmark).

## Building from source

Clone this repo using

```bash
git clone [url-to-repo] --recursive
```

Create a build tree using CMake, build and install:

```bash
cd your/checkout/folder/
cmake ..
make 
```
To build the documentation:

```bash
cd build/
make doc
```
### Dependencies

* CMake (with the [JRL CMake modules](https://github.com/jrl-umi3218/jrl-cmakemodules))
* Eigen>=3.0.5
* C++ >= 17

**Python dependencies (for some unit tests):**

* numpy
* scipy


