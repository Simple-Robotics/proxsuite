cmake_minimum_required(VERSION 3.10)

project(ExternalLib CXX)
find_package(proxsuite REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(run-proxqp ${CMAKE_CURRENT_LIST_DIR}/../src/run-proxqp.cpp)
target_link_libraries(run-proxqp PUBLIC proxsuite::proxsuite)
