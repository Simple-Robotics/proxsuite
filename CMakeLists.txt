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

SET(CMAKE_SKIP_RPATH FALSE)
LIST(APPEND CMAKE_BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
MESSAGE(STATUS "CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")
IF(APPLE) # Ensure that the policy if is only applied on OSX systems
  SET(CMAKE_MACOSX_RPATH TRUE)
  SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
     set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF("${isSystemDir}" STREQUAL "-1")
ENDIF(APPLE)

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
  SHARED
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

add_library(inria_ldlt_util STATIC test/src/util_f64.cpp test/src/util_f32.cpp)
target_include_directories(
  inria_ldlt_util PUBLIC test/include
)
target_link_libraries(inria_ldlt_util inria_ldlt)

set(CONAN_REQUIRES eigen/3.3.7 fmt/8.0.0)
if(ENABLE_TESTING)
  set(CONAN_REQUIRES
      ${CONAN_REQUIRES}
      # MIT License
#      backward-cpp/1.6
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
if(APPLE)
  set(CMAKE_SKIP_RPATH FALSE CACHE BOOL "rpaths" FORCE)
endif(APPLE)

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
  #pybind11_add_module(inria_ldlt_py src/ldlt_pybind11.cpp)
  add_library(inria_ldlt_py MODULE src/ldlt_pybind11.cpp)

  EXECUTE_PROCESS(
      COMMAND "${Python3_EXECUTABLE}" "-c"
      "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"
      OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)

  STRING(STRIP ${PYTHON_EXT_SUFFIX} PYTHON_EXT_SUFFIX)
  MESSAGE(STATUS "PYTHON_EXT_SUFFIX: ${PYTHON_EXT_SUFFIX}")
  target_link_libraries(inria_ldlt_py PUBLIC inria_ldlt pybind11::module)
  set_target_properties(inria_ldlt_py PROPERTIES
    BUILD_RPATH ${CMAKE_BUILD_RPATH}
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    )
endif()

if(ENABLE_TESTING)
  enable_testing()
#  add_subdirectory(test)
  add_subdirectory(benchmark)
endif()
