cmake_minimum_required(VERSION 3.22)

project(myproject)
find_package(proxsuite CONFIG REQUIRED COMPONENTS proxsuite vectorized)

add_executable(run-proxqp ${CMAKE_CURRENT_LIST_DIR}/../src/run-proxqp.cpp)
target_link_libraries(run-proxqp PUBLIC proxsuite::proxsuite-vectorized)
