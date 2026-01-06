cmake_minimum_required(VERSION 3.22)
project(myproject)

cmake_path(
  CONVERT
  "${PROXSUITE_FETCHCONTENT_SOURCE_DIR}"
  TO_CMAKE_PATH
  PROXSUITE_FETCHCONTENT_SOURCE_DIR
  NORMALIZE
)
message(STATUS "PROXSUITE_FETCHCONTENT_SOURCE_DIR: ${PROXSUITE_FETCHCONTENT_SOURCE_DIR}")

include(FetchContent)
FetchContent_Declare(proxsuite SOURCE_DIR ${PROXSUITE_FETCHCONTENT_SOURCE_DIR})
FetchContent_MakeAvailable(proxsuite)

add_executable(run-proxqp ${CMAKE_CURRENT_LIST_DIR}/../src/run-proxqp.cpp)
target_link_libraries(run-proxqp PUBLIC proxsuite)
