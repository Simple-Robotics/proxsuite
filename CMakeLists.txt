#
# Copyright (c) 2022 INRIA
#

cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME proxsuite)
set(PROJECT_DESCRIPTION "The proximal solver suite")
set(PROJECT_URL "http://github.com/Simple-Robotics/proxsuite")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)

# Set CMake Policies
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
endif(POLICY CMP0068)

# Check if the submodule cmake have been initialized
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake-module/base.cmake")
  message(
    FATAL_ERROR
      "\nPlease run the following command first:\ngit submodule update --init\n"
  )
endif()

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/apple.cmake)
include(CMakeDependentOption)

# If needed, set CMake policy for APPLE systems
apply_default_apple_configuration()

option(BUILD_PYTHON_INTERFACE "Build the Python bindings" OFF)
option(INSTALL_DOCUMENTATION "Generate and install the C++ documentation" OFF)
option(INITIALIZE_EIGEN_WITH_NAN "Initializa Eigen objects with NAN values" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)

option(BUILD_WITH_VECTORIZATION_SUPPORT
       "Build the library with the support of modern SIMD instructions." ON
)
option(BUILD_BINDINGS_WITH_AVX2_SUPPORT "Build the bindings with AVX2 support."
       ON
)
option(BUILD_BINDINGS_WITH_AVX512_SUPPORT
       "Build the bindings with AVX512 support." ON
)

# set(CMAKE_CXX_FLAGS "${CXX_COMPILATION_FLAGS} -march=native -mavx2 -fPIC
# -fno-omit-frame-pointer" )
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-external"
                      ${CMAKE_MODULE_PATH}
)

if(INITIALIZE_EIGEN_WITH_NAN)
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_EIGEN_WITH_NAN)

# Handle Windows context
if(MSVC)
  # add_definitions(-D_USE_MATH_DEFINES)
  add_definitions(-DNOMINMAX)
  CHECK_MINIMAL_CXX_STANDARD(20 ENFORCE)
else()
  CHECK_MINIMAL_CXX_STANDARD(17 ENFORCE)
endif()

# Look for dependencies
add_project_dependency(
  Eigen3
  REQUIRED
  PKG_CONFIG_REQUIRES
  "eigen3 >= 3.0.5"
)
if(BUILD_WITH_VECTORIZATION_SUPPORT)
  add_project_dependency(Simde REQUIRED)
endif()

# Build the main library
file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS ${PROJECT_SOURCE_DIR}/include/*.hpp)

add_library(proxsuite INTERFACE ${${PROJECT_NAME}_HEADERS})
target_link_libraries(
  proxsuite
  PUBLIC
  INTERFACE Eigen3::Eigen
)
set(EXPORTED_TARGETS_LIST proxsuite)

add_header_group(${PROJECT_NAME}_HEADERS)

add_library(proxsuite-not-vectorized INTERFACE)
target_link_libraries(
  proxsuite-not-vectorized
  PUBLIC
  INTERFACE proxsuite
)
target_compile_definitions(
  proxsuite-not-vectorized
  PUBLIC
  INTERFACE PROXSUITE_DONT_VECTORIZE
)
list(APPEND EXPORTED_TARGETS_LIST proxsuite-not-vectorized)

if(BUILD_WITH_VECTORIZATION_SUPPORT)
  add_library(proxsuite-vectorized INTERFACE)
  target_link_libraries(
    proxsuite-vectorized
    PUBLIC
    INTERFACE proxsuite
  )
  target_link_libraries(
    proxsuite-vectorized
    PUBLIC
    INTERFACE simde
  )
  list(APPEND EXPORTED_TARGETS_LIST proxsuite-vectorized)
endif()

install(
  TARGETS ${EXPORTED_TARGETS_LIST}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
)

# Activate only for advanced users to profile and benchmark the code
if(ADVANCED_USERS)
  include(cmake/compiler_warnings.cmake)
  include(cmake/extra_local_settings.cmake)
  include(cmake/static_analyzers.cmake)
  include(cmake/sanitizers.cmake)

  target_compile_definitions(proxsuite INTERFACE EIGEN_RUNTIME_NO_MALLOC)
  target_compile_definitions(
    proxsuite INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN
  )
  add_library(project_warnings INTERFACE)
  add_library(project_options INTERFACE)

  target_link_libraries(proxsuite INTERFACE project_options project_warnings)

  set_project_warnings(project_warnings)
  enable_sanitizers(project_options)
endif()

add_subdirectory(bindings)
add_subdirectory(test)
add_subdirectory(examples)
