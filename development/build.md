# Build and develop with pixi

The easiest way to set up a development environment is to use [pixi](https://pixi.sh/latest/#installation).

[pixi](https://pixi.sh/latest/) is a cross-platform package manager for developers.
It installs all required dependencies in the `.pixi` directory.
It's used by our CI, so you get the same stable and tested dependencies.

Run the following command to install dependencies, configure, build and test the project:

```bash
pixi run test
```

The project is built in the `build` directory.

The typical workflow is:

```bash
pixi shell 
pixi run configure
ninja -C build
```

After `pixi run configure`, use `cmake` and `ninja` manually to reconfigure and build the project.

## Environments

The pixi manifest contains many environments. The most common ones are:

- **default**: core proxsuite
- **all**: all proxsuite features

To activate a specific environment, run:

```bash
pixi shell -e all
```

Using **all** makes it easy to choose which features to build.
In this case, use the following CMake options:
- `BUILD_WITH_CHOLMOD_SUPPORT` : Build ProxSuite with the Cholmod support
- `BUILD_WITH_ACCELERATE_SUPPORT` : Build ProxSuite with the Accelerate support
With the **all** environment, all these options are ON.
To turn one off, pass the corresponding `-D` flag to `cmake`:

```bash
cmake -B build -DGENERATE_PYTHON_STUBS=OFF
```

## Faster build

When you work on a single feature with one associated test,build and run the corresponding test:
```bash
ninja -C build proxsuite-test-cpp-<name>
ctest --test-dir build --output-on-failure -R proxsuite-test-cpp-<name>
```
