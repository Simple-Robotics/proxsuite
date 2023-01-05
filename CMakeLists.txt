#
# Copyright (c) 2022 INRIA
#

cmake_minimum_required(VERSION 3.10)

if(DEFINED PROJECT_NAME)
  set(PROXSUITE_AS_SUBPROJECT)
endif()

set(PROJECT_NAME proxsuite)
set(PROJECT_DESCRIPTION "The Advanced Proximal Optimization Toolbox")
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
include(${CMAKE_CURRENT_LIST_DIR}/cmake-module/julia.cmake)
include(CMakeDependentOption)

# If needed, set CMake policy for APPLE systems
apply_default_apple_configuration()

option(BUILD_PYTHON_INTERFACE "Build the Python bindings" OFF)
option(INITIALIZE_EIGEN_WITH_NAN "Initialize Eigen objects with NAN values" OFF)
option(CHECK_RUNTIME_MALLOC
       "Check if some memory allocations are performed at runtime" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)

option(BUILD_WITH_VECTORIZATION_SUPPORT
       "Build the library with the support of modern SIMD instructions." ON)
option(BUILD_BINDINGS_WITH_AVX2_SUPPORT "Build the bindings with AVX2 support."
       ON)
option(BUILD_BINDINGS_WITH_AVX512_SUPPORT
       "Build the bindings with AVX512 support." ON)
option(TEST_JULIA_INTERFACE "Run the julia examples as unittest" OFF)

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/cmake-module/find-external/Julia"
    ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake-external"
                      ${CMAKE_MODULE_PATH})

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(INITIALIZE_EIGEN_WITH_NAN)
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_EIGEN_WITH_NAN)
if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_definitions(-DPROXSUITE_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

# set CXX standard
if(DEFINED CMAKE_CXX_STANDARD)
  check_minimal_cxx_standard(14 ENFORCE)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()

# Handle Windows context
if(MSVC)
  # add_definitions(-D_USE_MATH_DEFINES)
  add_definitions(-DNOMINMAX)
endif()

# Look for dependencies
add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")

set(SIMDE_HINT_FAILURE
    "Set BUILD_WITH_VECTORIZATION_SUPPORT=OFF or install Simde on your system.\n If Simde is already installed, ensure that the CMake variable CMAKE_MODULE_PATH correctly points toward the location of FindSimde.cmake file."
)
if(BUILD_WITH_VECTORIZATION_SUPPORT)
  add_project_dependency(Simde REQUIRED FIND_EXTERNAL "Simde"
                         PKG_CONFIG_REQUIRES "simde")
endif()

# Build the main library
file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS ${PROJECT_SOURCE_DIR}/include/*.hpp)

add_library(proxsuite INTERFACE)
if(MSVC)
  target_compile_options(proxsuite INTERFACE /permissive-)
  target_compile_options(proxsuite INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif(MSVC)
target_link_libraries(
  proxsuite
  PUBLIC
  INTERFACE Eigen3::Eigen)
target_include_directories(
  proxsuite INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_include_directories(
  proxsuite INTERFACE "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>")
set(EXPORTED_TARGETS_LIST proxsuite)

add_header_group(${PROJECT_NAME}_HEADERS)

if(BUILD_WITH_VECTORIZATION_SUPPORT)
  add_library(proxsuite-vectorized INTERFACE)
  target_link_libraries(
    proxsuite-vectorized
    PUBLIC
    INTERFACE proxsuite)
  target_link_libraries(
    proxsuite-vectorized
    PUBLIC
    INTERFACE simde)
  target_compile_definitions(proxsuite-vectorized INTERFACE PROXSUITE_VECTORIZE)
  list(APPEND EXPORTED_TARGETS_LIST proxsuite-vectorized)
endif()

if(BUILD_TESTING OR BUILD_PYTHON_INTERFACE)
  # Download cereal for pything bindings and unittests
  set(cereal_dir ${PROJECT_SOURCE_DIR}/external/cereal)
  set(cereal ${cereal_dir}/README.md)
  find_package(Git REQUIRED)
  if(NOT EXISTS ${cereal})
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init ${cereal_dir}
      WORKING_DIRECTORY ${cereal_dir} COMMAND_ERROR_IS_FATAL ANY)
  endif()
endif()

if(NOT PROXSUITE_AS_SUBPROJECT)
  install(
    TARGETS ${EXPORTED_TARGETS_LIST}
    EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})
endif(NOT PROXSUITE_AS_SUBPROJECT)

add_subdirectory(bindings)
if(BUILD_TESTING AND NOT PROXSUITE_AS_SUBPROJECT)
  add_subdirectory(test)
  add_subdirectory(examples)
endif()
