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
git clone https://github.com/Simple-Robotics/proxsuite.git --recursive
```

2. Create a build tree using CMake, build and install:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make
make install
```

Note: if you are building Proxsuite within a conda environment, consider passing `-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX`.

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

#### Disabling vectorization

We highly encourage you to enable the vectorization of the underlying linear algebra for the best performances.
They are active by default in **ProxSuite**.
Yet, some CPU architectures may not support such operations.
You just need to deactivate the cmake option `BUILD_WITH_VECTORIZATION_SUPPORT=OFF`, like:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_WITH_VECTORIZATION_SUPPORT=OFF
make
make install
```

#### Testing

To test the whole framework, you need installing first [Matio](https://github.com/tbeu/matio) (for reading .mat files in C++). You can then activate the build of the unit tests by activating the cmake option `BUILD_TESTING=ON`.
