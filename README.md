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


## Installation

### Building from source

#### Required install dependencies

The following dependencies are required at compile time:

* CMake
* Eigen >= 3.0.5
* C++ >= 17

#### Installation instructions

1. Clone this repository with:

```bash
git clone https://github.com/Simple-Robotics/proxqp.git --recursive
```

2. Create a build tree using CMake, build and install:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make 
make install
```

3. Build the Python interface

You just need to ensure that Python3 is indeed present on your system and activate the cmake option `BUILD_PYTHON_INTERFACE=ON` by replacing:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_PYTHON_INTERFACE=ON
make 
make install
```

4. Generate the doc

To build the documentation, you need installing Doxygen. Once it is done, it then is as simple as:

```bash
make doc
open doc/doxygen_html/index.html
```

#### Enabling vectorization

We highly encourage you to enable the vectorization of the underlying linear algebra for the best performances.
You just need to activate the cmake option `BUILD_WITH_SIMD_SUPPORT=ON`, like:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_WITH_SIMD_SUPPORT=ON
make 
make install
```

#### Testing

To test the whole framework, you need installing first [Matio](https://github.com/ami-iit/matio-cpp) (for reading .mat files in C++). You can then activate the build of the unit tests by activating the cmake option `BUILD_TESTING=ON`.
