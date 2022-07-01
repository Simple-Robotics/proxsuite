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
  MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
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

include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/python.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/apple.cmake)
include(CMakeDependentOption)

# If needed, set CMake policy for APPLE systems
apply_default_apple_configuration()

option(BUILD_PYTHON_INTERFACE "Build the Python bindings" OFF)
option(INSTALL_DOCUMENTATION "Generate and install the C++ documentation" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS
#     "${CXX_COMPILATION_FLAGS} -march=native -mavx2 -fPIC -fno-omit-frame-pointer"
# )
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake-external" ${CMAKE_MODULE_PATH})

# Look for dependencies
add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
if(BUILD_TESTING)
  find_package(Matio REQUIRED)
endif()
set(OPTIONAL_BUILD_TESTS OFF)
add_subdirectory(external/optional)

# Build the main library
add_library(proxsuite INTERFACE)
target_include_directories(proxsuite INTERFACE external/simde/)
target_include_directories(
  proxsuite INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/
)
target_link_libraries(proxsuite INTERFACE tl::optional)
target_link_libraries(proxsuite INTERFACE Eigen3::Eigen)

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

if(BUILD_PYTHON_INTERFACE)
  FINDPYTHON(REQUIRED)
endif()

add_subdirectory(bindings)
add_subdirectory(test)
#add_subdirectory(doc)
