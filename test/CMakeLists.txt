include(../cmake/doctest.cmake)

include_directories(include)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
add_library(libcnpy OBJECT src/cnpy.cpp)
target_link_libraries(libcnpy CONAN_PKG::eigen)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)

set(LDLT_TEST_LIBS
    inria_ldlt
    backward_cpp_main
    libcnpy
    inria_ldlt_util
    CONAN_PKG::benchmark
    CONAN_PKG::matio
)

macro(ldlt_test name path)
  add_executable(${name} ${path})
  doctest_discover_tests(${name})
  target_link_libraries(${name} ${LDLT_TEST_LIBS} doctest_main)
  set(LDLT_TEST_LIST ${LDLT_TEST_LIST} ${name})
endmacro()

macro(ldlt_bench name path)
  add_executable(${name} ${path})
  target_link_libraries(${name} ${LDLT_TEST_LIBS})
endmacro()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qp_problem/source_files.txt
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/save_qp_eq.py
          ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY src/save_qp_eq.py
)
add_custom_target(
  test.qp_eq.py.generated_problems
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qp_problem/source_files.txt
)
add_executable(test.qp_eq.py src/qp_eq_py.cpp)
target_link_libraries(test.qp_eq.py ${LDLT_TEST_LIBS})
add_dependencies(test.qp_eq.py test.qp_eq.py.generated_problems)

ldlt_test(test.precond.ruiz src/precond_ruiz.cpp)
ldlt_test(test.qp_eq.cpp src/qp_eq.cpp)
ldlt_test(test.qp_with_eq_and_in.cpp src/qp_with_eq_and_in.cpp)
ldlt_test(test.maros_meszaros src/test_maros_meszaros.cpp)
ldlt_test(test.sparse.ruiz src/test_ruiz_sparse.cpp)
ldlt_test(test.sparse.qp src/test_qp_sparse.cpp)
ldlt_test(test.sparse.maros_meszaros src/test_maros_meszaros_sparse.cpp)

ldlt_bench(bench.maros_meszaros src/bench_maros_meszaros.cpp)
ldlt_bench(bench.sparse.maros_meszaros src/bench_maros_meszaros_sparse.cpp)

target_compile_definitions(
  bench.maros_meszaros PUBLIC PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
  test.maros_meszaros PUBLIC PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
  test.sparse.maros_meszaros PUBLIC PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
  bench.sparse.maros_meszaros PUBLIC PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
  test.qp_eq.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${CMAKE_CURRENT_BINARY_DIR}/qp_problem/"
)
add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS ${LDLT_TEST_LIST}
)
