include(../cmake/doctest.cmake)

include_directories(include)

target_include_directories(
  inria_ldlt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
add_library(libcnpy OBJECT src/cnpy.cpp)
add_library(libinria_ldlt_util STATIC src/util_f64.cpp src/util_f32.cpp)
target_link_libraries(libinria_ldlt_util inria_ldlt)
target_link_libraries(libcnpy CONAN_PKG::eigen)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(LDLT_TEST_LIBS
    inria_ldlt
    backward_cpp_main
    doctest_main
    libcnpy
    libinria_ldlt_util
)

macro(ldlt_test testname testpath)
  add_executable(${testname} ${testpath})
  doctest_discover_tests(${testname})
  target_link_libraries(${testname} ${LDLT_TEST_LIBS})
  set(LDLT_TEST_LIST ${LDLT_TEST_LIST} ${testname})
endmacro()

ldlt_test(.test.permute.compute src/permutation_compute.cpp)
ldlt_test(.test.permute.apply src/permutation_apply.cpp)
ldlt_test(.test.permute.insert src/ldlt_insert.cpp)
ldlt_test(.test.permute.delete src/ldlt_delete.cpp)
ldlt_test(.test.factorize.roundtrip src/ldlt_roundtrip.cpp)
ldlt_test(.test.solve src/solve.cpp)
ldlt_test(.test.permute.solve src/permutation_solve.cpp)
ldlt_test(.test.update.roundtrip src/update_roundtrip.cpp)
ldlt_test(.test.mu_update.roundtrip src/mu_update_roundtrip.cpp)
ldlt_test(.test.delete src/delete.cpp)
ldlt_test(.test.append src/append.cpp)
ldlt_test(.test.ldlt_wrapper.solve src/ldlt_wrapper_solve.cpp)

ldlt_test(.test.precond.ruiz src/precond_ruiz.cpp)
ldlt_test(.test.qp_eq.cpp src/qp_eq.cpp)

add_executable(.test.add_with_permut.cpp src/row_add_with_permut.cpp)
doctest_discover_tests(.test.add_with_permut.cpp)
target_link_libraries(.test.add_with_permut.cpp ${LDLT_TEST_LIBS})

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qp_problem/source_files.txt
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/save_qp_eq.py
          ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY src/save_qp_eq.py
)
add_custom_target(
  .test.qp_eq.py.generated_problems
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qp_problem/source_files.txt
)
add_executable(.test.qp_eq.py src/qp_eq_py.cpp)
target_link_libraries(.test.qp_eq.py ${LDLT_TEST_LIBS})
set(INRIA_LDLT_QP_PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}/qp_problem/")
set(INRIA_LDLT_QP_PYTHON_PATH
    ${INRIA_LDLT_QP_PYTHON_PATH}
    PARENT_SCOPE
)
target_compile_definitions(
  .test.qp_eq.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${INRIA_LDLT_QP_PYTHON_PATH}"
)
add_dependencies(.test.qp_eq.py .test.qp_eq.py.generated_problems)

add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS ${LDLT_TEST_LIST}
)
