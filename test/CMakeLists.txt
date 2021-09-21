include(../cmake/doctest.cmake)

include_directories(include)

target_include_directories(
  inria_ldlt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
add_library(libcnpy OBJECT src/cnpy.cpp)
add_library(libinria_ldlt_util STATIC src/util_f64.cpp src/util_f32.cpp)
target_link_libraries(libinria_ldlt_util Eigen3::Eigen inria_ldlt)
target_link_libraries(libcnpy Eigen3::Eigen)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(LDLT_TEST_LIBS
    inria_ldlt
    backward_cpp_main
    doctest_main
    libcnpy
    libinria_ldlt_util
)

add_executable(.test.permute.compute src/permutation_compute.cpp)
doctest_discover_tests(.test.permute.compute)
target_link_libraries(.test.permute.compute ${LDLT_TEST_LIBS})

add_executable(.test.permute.apply src/permutation_apply.cpp)
doctest_discover_tests(.test.permute.apply)
target_link_libraries(.test.permute.apply ${LDLT_TEST_LIBS})

add_executable(.test.factorize.roundtrip src/ldlt_roundtrip.cpp)
doctest_discover_tests(.test.factorize.roundtrip)
target_link_libraries(.test.factorize.roundtrip ${LDLT_TEST_LIBS})

add_executable(.test.solve src/solve.cpp)
doctest_discover_tests(.test.solve)
target_link_libraries(.test.solve ${LDLT_TEST_LIBS})

add_executable(.test.permute.solve src/permutation_solve.cpp)
doctest_discover_tests(.test.permute.solve)
target_link_libraries(.test.permute.solve ${LDLT_TEST_LIBS})

add_executable(.test.update.roundtrip src/update_roundtrip.cpp)
doctest_discover_tests(.test.update.roundtrip)
target_link_libraries(.test.update.roundtrip ${LDLT_TEST_LIBS})

add_executable(.test.mu_update.roundtrip src/mu_update_roundtrip.cpp)
doctest_discover_tests(.test.mu_update.roundtrip)
target_link_libraries(.test.mu_update.roundtrip ${LDLT_TEST_LIBS})

add_executable(.test.row_delete src/row_delete.cpp)
doctest_discover_tests(.test.row_delete)
target_link_libraries(.test.row_delete ${LDLT_TEST_LIBS})

add_executable(.test.row_append src/row_append.cpp)
doctest_discover_tests(.test.row_append)
target_link_libraries(.test.row_append ${LDLT_TEST_LIBS})

add_executable(.test.precond.ruiz src/precond_ruiz.cpp)
doctest_discover_tests(.test.precond.ruiz)
target_link_libraries(.test.precond.ruiz ${LDLT_TEST_LIBS})

add_executable(.test.qp_eq.cpp src/qp_eq.cpp)
doctest_discover_tests(.test.qp_eq.cpp)
target_link_libraries(.test.qp_eq.cpp ${LDLT_TEST_LIBS})

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
  DEPENDS .test.permute.compute
          .test.permute.apply
          .test.permute.solve
          .test.factorize.roundtrip
          .test.solve
          .test.update.roundtrip
          .test.mu_update.roundtrip
          .test.row_delete
          .test.row_append
          .test.precond.ruiz
          .test.qp_eq.cpp
)
