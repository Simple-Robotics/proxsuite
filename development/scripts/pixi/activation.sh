#! /bin/bash
# Activation script

if [[ $PIXI_ENVIRONMENT_PLATFORMS == *"linux"* ]];
then
  # Conda compiler is named x86_64-conda-linux-gnu-c++, ccache can't resolve it
  # (https://ccache.dev/manual/latest.html#config_compiler_type)
  export CCACHE_COMPILERTYPE=gcc
fi
# Without -isystem, some LSP can't find headers
export PROXSUITE_CXX_FLAGS="$PROXSUITE_CXX_FLAGS -isystem $CONDA_PREFIX/include"

# Set default build value only if not previously set
export PROXSUITE_BUILD_TYPE=${PROXSUITE_BUILD_TYPE:=Release}
export PROXSUITE_BUILD_VECTORIZATION=${PROXSUITE_BUILD_VECTORIZATION:=ON}
export PROXSUITE_BUILD_PYTHON_INTERFACE=${PROXSUITE_BUILD_PYTHON_INTERFACE:=OFF}
export PROXSUITE_BUILD_TESTING=${PROXSUITE_BUILD_TESTING:=ON}
export PROXSUITE_BUILD_MAROS_MESZAROS_TESTS=${PROXSUITE_BUILD_MAROS_MESZAROS_TESTS:=OFF}
