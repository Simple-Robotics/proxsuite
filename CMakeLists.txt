# Copyright (c) 2022-2026 Inria

cmake_minimum_required(VERSION 3.22...4.2)

project(
  proxsuite
  VERSION 0.7.2
  DESCRIPTION "The Advanced Proximal Optimization Toolbox"
  HOMEPAGE_URL "https://github.com/Simple-Robotics/proxsuite"
)

include(cmake/get-jrl-cmakemodules.cmake)

jrl_configure_defaults()

jrl_option(BUILD_TESTING "Build the tests" ON)
jrl_option(BUILD_EXAMPLES "Build the examples" OFF)
jrl_option(BUILD_BENCHMARK "Build the benchmarks" OFF)
jrl_option(BUILD_DOCUMENTATION "Build the documentation." OFF)
jrl_option(BUILD_MAROS_MESZAROS_TESTS "Build the Maros-Meszaros tests, needs matio library" OFF)
jrl_option(ENABLE_WARNINGS "Enable warnings during compilation" OFF)
jrl_option(ENABLE_WARNINGS_AS_ERRORS "Treat all warnings as errors" OFF)
jrl_option(BUILD_PYTHON_INTERFACE "Build the Python bindings" OFF)
jrl_option(INSTALL_DOCUMENTATION "Install the documentation" OFF)
jrl_option(INITIALIZE_EIGEN_WITH_NAN "Initialize Eigen objects with NAN values" OFF)
jrl_option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)
jrl_option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
jrl_option(BUILD_WITH_VECTORIZATION_SUPPORT "Build the library with the support of modern SIMD instructions." OFF)
jrl_cmake_dependent_option(BUILD_BINDINGS_WITH_AVX2_SUPPORT "Build the bindings with AVX2 support." ON BUILD_WITH_VECTORIZATION_SUPPORT OFF)
jrl_cmake_dependent_option(BUILD_BINDINGS_WITH_AVX512_SUPPORT "Build the bindings with AVX512 support." ON BUILD_WITH_VECTORIZATION_SUPPORT OFF)
jrl_option(TEST_JULIA_INTERFACE "Run the julia examples as unittest" OFF)
jrl_option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" OFF)
jrl_cmake_dependent_option(
  LINK_PYTHON_INTERFACE_TO_OPENMP
  "Link OpenMP to the Python interface"
  ON
  BUILD_WITH_OPENMP_SUPPORT
  OFF
)
jrl_cmake_dependent_option(BUILD_WITH_SERIALIZATION "Build with serialization support" ON BUILD_PYTHON_INTERFACE OFF)
jrl_option(GENERATE_PYTHON_STUBS "Generate Python stubs" OFF)
jrl_option(PYTHON_BINDING_CUSTOM_INSTALL_DIR "Override the default <current-python-install-sitelib>/<project-name>" "")
jrl_option(PROXSUITE_DOXYGEN_USE_MATHJAX "Avoid latex at build time" OFF)

############################################################

# Eigen3
jrl_find_package(Eigen3 CONFIG REQUIRED)

# Catch2
if(BUILD_TESTING)
  jrl_find_package(Catch2 3.0 CONFIG QUIET)
  if(Catch2_FOUND)
    include(Catch)
  else()
    message(
      WARNING
      "[proxsuite] Catch2 version >= 3.0 is required for testing.
      Falling back to FetchContent to get Catch2 v3.11.0"
    )
    include(FetchContent)
    FetchContent_Declare(
      catch2
      URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.11.0.zip
      URL_HASH MD5=ce96600812f04d19694f488ed2de75c5
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(catch2)
    include(${catch2_SOURCE_DIR}/extras/Catch.cmake)
  endif()
endif()

# cereal (Serialization)
if(BUILD_WITH_SERIALIZATION)
  jrl_find_package(cereal CONFIG QUIET)
  if(NOT cereal_FOUND)
    message(
      WARNING
      "[proxsuite] cereal library not found. Falling back to FetchContent to get cereal v1.3.2"
    )
    include(FetchContent)
    FetchContent_Declare(
      cereal
      URL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.zip
      URL_HASH MD5=953a872cd3d78abf2e29212987a0ba71
      SOURCE_SUBDIR
      download_only
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(cereal)
    add_library(cereal::cereal INTERFACE IMPORTED)
    target_include_directories(cereal::cereal SYSTEM INTERFACE ${cereal_SOURCE_DIR}/include)
  endif()
endif()

# nanobind (Python bindings)
if(BUILD_PYTHON_INTERFACE)
  jrl_find_python(3.8 REQUIRED COMPONENTS Interpreter Development.Module)
  jrl_find_nanobind(2.4.0 CONFIG QUIET)
  if(NOT nanobind_FOUND)
    message(
      WARNING
      "[proxsuite] nanobind library not found. Falling back to FetchContent to get nanobind v2.9.2"
    )
    include(FetchContent)
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG v2.9.2
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(nanobind)
  endif()
endif()

# simde
if(BUILD_WITH_VECTORIZATION_SUPPORT)
  jrl_find_package(simde REQUIRED)
endif()

# matio
if(BUILD_TESTING AND BUILD_MAROS_MESZAROS_TESTS)
  jrl_find_package(matio REQUIRED)
endif()

# OpenMP
if(BUILD_WITH_OPENMP_SUPPORT)
  jrl_find_package(OpenMP REQUIRED)
endif()

# Doxygen + Graphviz (dot)
if(BUILD_DOCUMENTATION)
  jrl_find_package(Doxygen REQUIRED COMPONENTS dot)
endif()

############################################################

add_library(proxsuite INTERFACE)
add_library(proxsuite::proxsuite ALIAS proxsuite)
set_target_properties(proxsuite PROPERTIES VERSION ${PROJECT_VERSION})
target_compile_features(proxsuite INTERFACE cxx_std_17)
jrl_target_enforce_msvc_conformance(proxsuite INTERFACE)

target_include_directories(
  proxsuite
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

jrl_target_generate_config_header(proxsuite INTERFACE)

if(ENABLE_WARNINGS)
  jrl_target_set_default_compile_options(proxsuite INTERFACE)
endif()

if(ENABLE_WARNINGS_AS_ERRORS)
  jrl_target_treat_all_warnings_as_errors(proxsuite INTERFACE)
endif()

target_compile_definitions(
  proxsuite
  INTERFACE
    $<$<BOOL:${INITIALIZE_EIGEN_WITH_NAN}>:EIGEN_INITIALIZE_MATRICES_BY_NAN>
    $<$<BOOL:${CHECK_RUNTIME_MALLOC}>:EIGEN_RUNTIME_NO_MALLOC>
    $<$<BOOL:${CHECK_RUNTIME_MALLOC}>:PROXSUITE_EIGEN_CHECK_MALLOC>
    $<$<BOOL:${BUILD_WITH_OPENMP_SUPPORT}>:PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP>
)

target_link_libraries(
  proxsuite
  INTERFACE
    Eigen3::Eigen
    $<$<BOOL:${BUILD_WITH_OPENMP_SUPPORT}>:OpenMP::OpenMP_CXX>
    $<$<BOOL:${BUILD_WITH_SERIALIZATION}>:cereal::cereal>
)

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings/python)
endif()

if(BUILD_WITH_VECTORIZATION_SUPPORT)
  add_library(proxsuite-vectorized INTERFACE)
  add_library(proxsuite::proxsuite-vectorized ALIAS proxsuite-vectorized)
  target_link_libraries(
    proxsuite-vectorized
    INTERFACE proxsuite simde::simde $<$<BOOL:${BUILD_WITH_SERIALIZATION}>:cereal::cereal>
  )
  target_compile_definitions(proxsuite-vectorized INTERFACE PROXSUITE_VECTORIZE)
endif()

if(BUILD_TESTING)
  message(STATUS "[proxsuite] Building tests")
  enable_testing()
  add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
  message(STATUS "[proxsuite] Building examples")
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARK)
  message(STATUS "[proxsuite] Building benchmarks")
  add_subdirectory(benchmark)
endif()

if(BUILD_DOCUMENTATION)
  message(STATUS "[proxsuite] Building documentation")
  add_subdirectory(doc)
endif()

jrl_target_headers(proxsuite INTERFACE
  HEADERS
    include/proxsuite/helpers/common.hpp
    include/proxsuite/helpers/instruction-set.hpp
    include/proxsuite/helpers/optional.hpp
    include/proxsuite/helpers/version.hpp
    include/proxsuite/linalg/dense/core.hpp
    include/proxsuite/linalg/dense/factorize.hpp
    include/proxsuite/linalg/dense/ldlt.hpp
    include/proxsuite/linalg/dense/modify.hpp
    include/proxsuite/linalg/dense/solve.hpp
    include/proxsuite/linalg/dense/update.hpp
    include/proxsuite/linalg/sparse/core.hpp
    include/proxsuite/linalg/sparse/factorize.hpp
    include/proxsuite/linalg/sparse/rowmod.hpp
    include/proxsuite/linalg/sparse/update.hpp
    include/proxsuite/linalg/veg/internal/external/hedley.ext.hpp
    include/proxsuite/linalg/veg/internal/external/unhedley.ext.hpp
    include/proxsuite/linalg/veg/internal/assert_impl.hpp
    include/proxsuite/linalg/veg/internal/collection_algo.hpp
    include/proxsuite/linalg/veg/internal/dbg.hpp
    include/proxsuite/linalg/veg/internal/delete_special_members.hpp
    include/proxsuite/linalg/veg/internal/dyn_index.hpp
    include/proxsuite/linalg/veg/internal/epilogue.hpp
    include/proxsuite/linalg/veg/internal/fix_index.hpp
    include/proxsuite/linalg/veg/internal/has_asan.hpp
    include/proxsuite/linalg/veg/internal/integer_seq.hpp
    include/proxsuite/linalg/veg/internal/macros.hpp
    include/proxsuite/linalg/veg/internal/narrow.hpp
    include/proxsuite/linalg/veg/internal/preprocessor.hpp
    include/proxsuite/linalg/veg/internal/prologue.hpp
    include/proxsuite/linalg/veg/internal/std.hpp
    include/proxsuite/linalg/veg/internal/terminate.hpp
    include/proxsuite/linalg/veg/internal/typedefs.hpp
    include/proxsuite/linalg/veg/memory/address.hpp
    include/proxsuite/linalg/veg/memory/alloc.hpp
    include/proxsuite/linalg/veg/memory/dynamic_stack.hpp
    include/proxsuite/linalg/veg/memory/placement.hpp
    include/proxsuite/linalg/veg/memory/stack_alloc.hpp
    include/proxsuite/linalg/veg/type_traits/alloc.hpp
    include/proxsuite/linalg/veg/type_traits/assignable.hpp
    include/proxsuite/linalg/veg/type_traits/constructible.hpp
    include/proxsuite/linalg/veg/type_traits/core.hpp
    include/proxsuite/linalg/veg/type_traits/invocable.hpp
    include/proxsuite/linalg/veg/type_traits/primitives.hpp
    include/proxsuite/linalg/veg/type_traits/tags.hpp
    include/proxsuite/linalg/veg/util/assert.hpp
    include/proxsuite/linalg/veg/util/dbg.hpp
    include/proxsuite/linalg/veg/util/defer.hpp
    include/proxsuite/linalg/veg/util/dynstack_alloc.hpp
    include/proxsuite/linalg/veg/util/get.hpp
    include/proxsuite/linalg/veg/util/index.hpp
    include/proxsuite/linalg/veg/util/unreachable.hpp
    include/proxsuite/linalg/veg/ref.hpp
    include/proxsuite/linalg/veg/slice.hpp
    include/proxsuite/linalg/veg/tuple.hpp
    include/proxsuite/linalg/veg/vec.hpp
    include/proxsuite/proxqp/dense/preconditioner/identity.hpp
    include/proxsuite/proxqp/dense/preconditioner/ruiz.hpp
    include/proxsuite/proxqp/dense/backward_data.hpp
    include/proxsuite/proxqp/dense/compute_ECJ.hpp
    include/proxsuite/proxqp/dense/dense.hpp
    include/proxsuite/proxqp/dense/fwd.hpp
    include/proxsuite/proxqp/dense/helpers.hpp
    include/proxsuite/proxqp/dense/linesearch.hpp
    include/proxsuite/proxqp/dense/model.hpp
    include/proxsuite/proxqp/dense/solver.hpp
    include/proxsuite/proxqp/dense/utils.hpp
    include/proxsuite/proxqp/dense/views.hpp
    include/proxsuite/proxqp/dense/workspace.hpp
    include/proxsuite/proxqp/dense/wrapper.hpp
    include/proxsuite/proxqp/parallel/omp.hpp
    include/proxsuite/proxqp/parallel/qp_solve.hpp
    include/proxsuite/proxqp/sparse/preconditioner/identity.hpp
    include/proxsuite/proxqp/sparse/preconditioner/ruiz.hpp
    include/proxsuite/proxqp/sparse/fwd.hpp
    include/proxsuite/proxqp/sparse/helpers.hpp
    include/proxsuite/proxqp/sparse/model.hpp
    include/proxsuite/proxqp/sparse/solver.hpp
    include/proxsuite/proxqp/sparse/sparse.hpp
    include/proxsuite/proxqp/sparse/utils.hpp
    include/proxsuite/proxqp/sparse/views.hpp
    include/proxsuite/proxqp/sparse/workspace.hpp
    include/proxsuite/proxqp/sparse/wrapper.hpp
    include/proxsuite/proxqp/utils/prints.hpp
    include/proxsuite/proxqp/utils/random_qp_problems.hpp
    include/proxsuite/proxqp/results.hpp
    include/proxsuite/proxqp/settings.hpp
    include/proxsuite/proxqp/status.hpp
    include/proxsuite/proxqp/timings.hpp
    include/proxsuite/fwd.hpp
  BASE_DIRS include
)

if(BUILD_WITH_SERIALIZATION)
  jrl_target_headers(proxsuite INTERFACE
    HEADERS
      include/proxsuite/serialization/archive.hpp
      include/proxsuite/serialization/eigen.hpp
      include/proxsuite/serialization/model.hpp
      include/proxsuite/serialization/results.hpp
      include/proxsuite/serialization/ruiz.hpp
      include/proxsuite/serialization/settings.hpp
      include/proxsuite/serialization/workspace.hpp
      include/proxsuite/serialization/wrapper.hpp
    BASE_DIRS include
  )
endif()

jrl_add_export_component(NAME proxsuite TARGETS proxsuite)

if(BUILD_WITH_VECTORIZATION_SUPPORT)
  jrl_add_export_component(NAME vectorized TARGETS proxsuite-vectorized)
endif()

jrl_export_package()

jrl_print_dependencies_summary()
jrl_print_options_summary()
