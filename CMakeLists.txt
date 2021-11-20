cmake_minimum_required(VERSION 3.16)
project(inria_ldlt CXX)

get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
  set(top_level FALSE)
else()
  set(top_level TRUE)
endif()

option(INRIA_LDLT_ENABLE_TESTING "build with tests" OFF)

if(top_level AND INRIA_LDLT_ENABLE_TESTING)
  set(ENABLE_TESTING ON)
else()
  set(ENABLE_TESTING OFF)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(ENABLE_DOCS_ONLY)
  add_subdirectory("docs")
  return()
endif()

if(ENABLE_TESTING)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

include(cmake/compiler_warnings.cmake)
include(cmake/static_analyzers.cmake)
include(cmake/sanitizers.cmake)
include(cmake/conan.cmake)

if(ENABLE_TESTING)
  option(INRIA_LDLT_DEBUG "build with debug symbols" ON)
  if(INRIA_LDLT_DEBUG)
    add_compile_options("-g")
  endif()
  include(cmake/extra_local_settings.cmake)
endif()

add_library(
  inria_ldlt
  ${CMAKE_CURRENT_SOURCE_DIR}/src/eigen_f32.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/eigen_f64.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/factorize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/update.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solve.cpp
  #${CMAKE_CURRENT_SOURCE_DIR}/test/src/cnpy.cpp
)
target_compile_options(inria_ldlt PRIVATE "-march=native")
target_include_directories(
  inria_ldlt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/
)
target_include_directories(
  inria_ldlt SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/simde/
)

set(CONAN_REQUIRES eigen/3.3.7 fmt/8.0.0)
if(ENABLE_TESTING)
  set(CONAN_REQUIRES
      ${CONAN_REQUIRES}
      # MIT License
      backward-cpp/1.5
      #
      # Apache-2.0
      benchmark/1.6.0
      #
      # LGPL v3
      mpfr/4.1.0
  )
  set(CONAN_OPTIONS)
endif(ENABLE_TESTING)
run_conan()

# Conan dependencies
if(ENABLE_TESTING)

  target_compile_definitions(inria_ldlt PUBLIC EIGEN_RUNTIME_NO_MALLOC)
  target_compile_definitions(inria_ldlt PUBLIC EIGEN_INITIALIZE_MATRICES_BY_NAN)
  add_subdirectory(external/osqp)
  add_subdirectory(external/qpalm)
  target_link_libraries(inria_ldlt PUBLIC CONAN_PKG::fmt CONAN_PKG::mpfr)

  find_package(Python3 COMPONENTS Interpreter)
  add_library(project_warnings INTERFACE)
  add_library(project_options INTERFACE)

  target_link_libraries(inria_ldlt PUBLIC project_options project_warnings)

  set_project_warnings(project_warnings)
  enable_sanitizers(project_options)
endif()

target_link_libraries(inria_ldlt PUBLIC CONAN_PKG::eigen CONAN_PKG::fmt)
set_property(TARGET inria_ldlt PROPERTY POSITION_INDEPENDENT_CODE ON)

if(INRIA_LDLT_PYTHON_BINDINGS)
  find_package(Python3 COMPONENTS Interpreter Development)
  add_subdirectory(external/pybind11)
  pybind11_add_module(inria_ldlt_py src/ldlt_pybind11.cpp)
  target_link_libraries(inria_ldlt_py PUBLIC inria_ldlt)
endif()

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(benchmark)
endif()
