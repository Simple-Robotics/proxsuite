cmake_minimum_required(VERSION 3.15)
project(prox-suite CXX)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

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
set(CMAKE_CXX_FLAGS
    "${CXX_COMPILATION_FLAGS} -march=native -mavx2 -fPIC -fno-omit-frame-pointer"
)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_SKIP_RPATH FALSE)
list(APPEND CMAKE_BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")
if(APPLE) # Ensure that the policy if is only applied on OSX systems
  set(CMAKE_MACOSX_RPATH TRUE)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  list(
    FIND
    CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_INSTALL_PREFIX}/lib"
    isSystemDir
  )
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
endif(APPLE)

if(ENABLE_DOCS_ONLY)
  add_subdirectory(doc)
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

add_library(prox-suite INTERFACE)
target_compile_options(prox-suite INTERFACE "-march=native")
target_include_directories(prox-suite INTERFACE external/simde/)
target_include_directories(
  prox-suite INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

add_library(prox-suite-util STATIC test/src/util_f64.cpp test/src/util_f32.cpp)
target_include_directories(prox-suite-util PUBLIC test/include)
target_link_libraries(prox-suite-util prox-suite)

set(CONAN_REQUIRES eigen/3.3.7 fmt/8.0.0)
if(ENABLE_TESTING)
  set(CONAN_REQUIRES
      ${CONAN_REQUIRES}
      matio/1.5.21
      # MIT License
      backward-cpp/1.6
      #
      # Apache-2.0
      benchmark/1.6.0
  )
  set(CONAN_OPTIONS)
endif(ENABLE_TESTING)
run_conan()
if(APPLE)
  set(CMAKE_SKIP_RPATH
      FALSE
      CACHE BOOL "rpaths" FORCE
  )
endif(APPLE)

# Conan dependencies
if(ENABLE_TESTING)

  target_compile_definitions(prox-suite INTERFACE EIGEN_RUNTIME_NO_MALLOC)
  target_compile_definitions(
    prox-suite INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN
  )
  add_subdirectory(external/osqp)
  add_subdirectory(external/qpalm)
  target_link_libraries(prox-suite INTERFACE CONAN_PKG::fmt)

  find_package(Python3 COMPONENTS Interpreter)
  add_library(project_warnings INTERFACE)
  add_library(project_options INTERFACE)

  target_link_libraries(prox-suite INTERFACE project_options project_warnings)

  set_project_warnings(project_warnings)
  enable_sanitizers(project_options)
endif()

target_link_libraries(prox-suite INTERFACE CONAN_PKG::eigen CONAN_PKG::fmt)

add_subdirectory(external/veg)
target_link_libraries(prox-suite INTERFACE veg::veg)

if(INRIA_LDLT_PYTHON_BINDINGS)
  target_compile_definitions(
    prox-suite INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN
  )
  find_package(Python3 COMPONENTS Interpreter Development)
  add_subdirectory(external/pybind11)
  add_library(prox_suite_py MODULE src/ldlt_pybind11.cpp)

  execute_process(
    COMMAND
      "${Python3_EXECUTABLE}" "-c"
      "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
  )

  string(STRIP ${PYTHON_EXT_SUFFIX} PYTHON_EXT_SUFFIX)
  message(STATUS "PYTHON_EXT_SUFFIX: ${PYTHON_EXT_SUFFIX}")
  target_link_libraries(prox_suite_py PUBLIC prox-suite pybind11::module)
  set_target_properties(
    prox_suite_py
    PROPERTIES BUILD_RPATH ${CMAKE_BUILD_RPATH}
               PREFIX ""
               SUFFIX "${PYTHON_EXT_SUFFIX}"
  )
endif()

if(ENABLE_TESTING)
  set(EXTRA_CXX_FLAGS
      $ENV{EXTRA_CXX_FLAGS}
      CACHE STRING "extra compiler flags for development"
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(doc)
endif()
