cmake_minimum_required(VERSION 3.16)
project(inria_ldlt CXX)

get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
  set(top_level FALSE)
else()
  set(top_level TRUE)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(ENABLE_DOCS_ONLY)
  add_subdirectory("docs")
  return()
endif()

if(top_level AND INRIA_LDLT_ENABLE_TESTING)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

include(cmake/compiler_warnings.cmake)
include(cmake/static_analyzers.cmake)
include(cmake/sanitizers.cmake)
include(cmake/conan.cmake)

add_library(inria_ldlt INTERFACE)
target_include_directories(
  inria_ldlt SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/
                              ${CMAKE_CURRENT_SOURCE_DIR}/external/simde/
)

if(top_level AND INRIA_LDLT_ENABLE_TESTING)
  add_library(project_warnings INTERFACE)
  add_library(project_options INTERFACE)

  target_link_libraries(inria_ldlt INTERFACE project_options project_warnings)

  include(cmake/extra_local_settings.cmake)

  set_project_warnings(project_warnings)

  enable_sanitizers(project_options)

  enable_testing()
  add_subdirectory(test)
  add_subdirectory(benchmark)
endif()
