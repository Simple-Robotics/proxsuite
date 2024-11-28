include(../cmake-external/doctest.cmake)
find_package(Matio REQUIRED)

add_library(${PROJECT_NAME}-doctest STATIC doctest/doctest.cpp)
target_include_directories(${PROJECT_NAME}-doctest PUBLIC ./doctest)
add_library(cnpy OBJECT src/cnpy.cpp)
target_link_libraries(cnpy Eigen3::Eigen)
target_include_directories(cnpy PUBLIC ./include)

file(GLOB_RECURSE TEST_HEADERS ./include/*.hpp)
add_header_group(TEST_HEADERS)

add_library(proxsuite-test-util STATIC ./src/util_f64.cpp ./src/util_f32.cpp
                                       ${TEST_HEADERS})
target_include_directories(proxsuite-test-util PUBLIC ./include)
if(BUILD_WITH_VECTORIZATION_SUPPORT)
  target_link_libraries(proxsuite-test-util proxsuite-vectorized matio)
else()
  target_link_libraries(proxsuite-test-util proxsuite matio)
endif()

macro(proxsuite_test name path)
  set(target_name ${PROJECT_NAME}-test-cpp-${name})
  add_executable(${target_name} ${path})
  doctest_discover_tests(${target_name})
  target_link_libraries(${target_name} PUBLIC proxsuite ${PROJECT_NAME}-doctest
                                              proxsuite-test-util)
  target_compile_definitions(${target_name}
                             PRIVATE PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
  add_dependencies(build_tests ${target_name})
endmacro()

proxsuite_test(dense_ruiz_equilibration src/dense_ruiz_equilibration.cpp)
proxsuite_test(dense_qp_eq src/dense_qp_eq.cpp)
proxsuite_test(dense_qp_with_eq_and_in src/dense_qp_with_eq_and_in.cpp)
proxsuite_test(dense_qp_unconstrained src/dense_unconstrained_qp.cpp)
proxsuite_test(dense_backward src/dense_backward.cpp)
proxsuite_test(dense_qp_wrapper src/dense_qp_wrapper.cpp)
proxsuite_test(dense_qp_solve src/dense_qp_solve.cpp)
proxsuite_test(sparse_ruiz_equilibration src/sparse_ruiz_equilibration.cpp)
proxsuite_test(sparse_qp src/sparse_qp.cpp)
proxsuite_test(sparse_qp_wrapper src/sparse_qp_wrapper.cpp)
proxsuite_test(sparse_qp_solve src/sparse_qp_solve.cpp)
proxsuite_test(sparse_factorization src/sparse_factorization.cpp)
proxsuite_test(cvxpy src/cvxpy.cpp)

if(BUILD_WITH_OPENMP_SUPPORT)
  proxsuite_test(parallel src/parallel_qp_solve.cpp)
  target_link_libraries(${PROJECT_NAME}-test-cpp-parallel
                        PRIVATE OpenMP::OpenMP_CXX)
endif()

# Test serialization
macro(ADD_TEST_CFLAGS target flag)
  set_property(
    TARGET ${target}
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " ${flag}")
endmacro()

make_directory("${CMAKE_CURRENT_BINARY_DIR}/serialization-data")
proxsuite_test(serialization src/serialization.cpp)
add_test_cflags(
  ${PROJECT_NAME}-test-cpp-serialization
  "-DTEST_SERIALIZATION_FOLDER=\\\\\"${CMAKE_CURRENT_BINARY_DIR}/serialization-data\\\\\""
)
if(cereal_FOUND)
  target_link_libraries(${PROJECT_NAME}-test-cpp-serialization PRIVATE cereal::cereal)
else()
  target_include_directories(
    ${PROJECT_NAME}-test-cpp-serialization SYSTEM
    PRIVATE ${PROJECT_SOURCE_DIR}/external/cereal/include)
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
  proxsuite_test(dense_maros_meszaros src/dense_maros_meszaros.cpp)
  proxsuite_test(sparse_maros_meszaros src/sparse_maros_meszaros.cpp)
endif()

if(BUILD_PYTHON_INTERFACE)
  file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_UNITTEST *.py)

  if((CMAKE_BUILD_TYPE STREQUAL "Debug" AND MSVC) OR NOT
                                                     BUILD_WITH_OPENMP_SUPPORT)
    list_filter("${${PROJECT_NAME}_PYTHON_UNITTEST}" "parallel_qp_solve"
                ${PROJECT_NAME}_PYTHON_UNITTEST)
  endif()

  foreach(TEST_FILE ${${PROJECT_NAME}_PYTHON_UNITTEST})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/" "" TEST_FILE ${TEST_FILE})
    add_python_unit_test("${PROJECT_NAME}-test-py-${TEST_NAME}" "${TEST_FILE}"
                         "bindings/python")
  endforeach()
endif(BUILD_PYTHON_INTERFACE)
