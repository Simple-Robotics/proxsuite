include(../cmake/doctest.cmake)

include_directories(include)

target_include_directories(
  inria_ldlt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
add_library(libcnpy OBJECT src/cnpy.cpp)
target_link_libraries(libcnpy Eigen3::Eigen)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(LDLT_TEST_LIBS
    inria_ldlt
    backward_cpp_main
    doctest_main
    libcnpy
)

add_executable(.test.factorize.roundtrip src/ldlt_roundtrip.cpp)
target_link_libraries(.test.factorize.roundtrip ${LDLT_TEST_LIBS})

add_executable(.test.solve src/solve.cpp)
target_link_libraries(.test.solve ${LDLT_TEST_LIBS})

add_executable(.test.update.roundtrip src/update_roundtrip.cpp)
target_link_libraries(.test.update.roundtrip ${LDLT_TEST_LIBS})

add_executable(.test.mu_update.roundtrip src/mu_update_roundtrip.cpp)
target_link_libraries(.test.mu_update.roundtrip ${LDLT_TEST_LIBS})

add_executable(.test.precond.ruiz src/precond_ruiz.cpp)
target_link_libraries(.test.precond.ruiz ${LDLT_TEST_LIBS})

add_executable(.test.qp_eq.cpp src/qp_eq.cpp)
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
set(INRIA_LDLT_QP_PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}/qp_problem/" PARENT_SCOPE)
target_compile_definitions(
  .test.qp_eq.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${INRIA_LDLT_QP_PYTHON_PATH}"
)
add_dependencies(.test.qp_eq.py .test.qp_eq.py.generated_problems)
