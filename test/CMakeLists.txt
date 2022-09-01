include(../cmake-external/doctest.cmake)
find_package(Matio REQUIRED)

add_library(doctest STATIC doctest/doctest.cpp)
target_include_directories(doctest PUBLIC ./doctest)
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
  add_executable(test-cpp-${name} ${path})
  doctest_discover_tests(test-cpp-${name})
  target_link_libraries(test-cpp-${name} proxsuite doctest proxsuite-test-util)
  target_compile_definitions(test-cpp-${name}
                             PRIVATE PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
endmacro()

if(0)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qp_problem/source_files.txt
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/save_qp_eq.py
            ${CMAKE_CURRENT_BINARY_DIR}
    MAIN_DEPENDENCY src/save_qp_eq.py)
  add_custom_target(
    test.qp_eq.py.generated_problems
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qp_proble m/source_files.txt)

  add_executable(test.qp_eq.py src/qp_eq_py.cpp)
  add_dependencies(test.qp_eq.py test.qp_eq.py.generated_problems)
  target_link_libraries(test.qp_eq.py ${LDLT_TEST_LIBS} doctest)

  target_compile_definitions(
    test.qp_eq.py
    PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${CMAKE_CURRENT_BINARY_DIR}/qp_problem/")
endif()

proxsuite_test(dense_ruiz_equilibration src/dense_ruiz_equilibration.cpp)
proxsuite_test(dense_qp_eq src/dense_qp_eq.cpp)
proxsuite_test(dense_qp_with_eq_and_in src/dense_qp_with_eq_and_in.cpp)
proxsuite_test(dense_qp_unconstrained src/dense_unconstrained_qp.cpp)
proxsuite_test(dense_maros_meszaros src/dense_maros_meszaros.cpp)
proxsuite_test(sparse_ruiz_equilibration src/sparse_ruiz_equilibration.cpp)
proxsuite_test(sparse_qp src/sparse_qp.cpp)
proxsuite_test(sparse_maros_meszaros src/sparse_maros_meszaros.cpp)
proxsuite_test(dense_qp_wrapper src/dense_qp_wrapper.cpp)
proxsuite_test(dense_qp_solve src/dense_qp_solve.cpp)
proxsuite_test(sparse_qp_wrapper src/sparse_qp_wrapper.cpp)
proxsuite_test(sparse_qp_solve src/sparse_qp_solve.cpp)
proxsuite_test(sparse_factorization src/sparse_factorization.cpp)
proxsuite_test(cvxpy src/cvxpy.cpp)

if(BUILD_PYTHON_INTERFACE)
  file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_UNITTEST *.py)
  list(REMOVE_ITEM ${PROJECT_NAME}_PYTHON_UNITTEST ${CMAKE_CURRENT_SOURCE_DIR}/src/save_qp_eq.py)
  foreach(TEST ${${PROJECT_NAME}_PYTHON_UNITTEST})
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/test/src/" "" TEST ${TEST})
    add_python_unit_test("py-${TEST}" "test/src/${TEST}")
  endforeach(TEST ${${PROJECT_NAME}_PYTHON_UNITTEST})
endif(BUILD_PYTHON_INTERFACE)