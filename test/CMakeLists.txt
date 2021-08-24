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

add_executable(.test.qp_eq src/qp_eq.cpp)
target_link_libraries(.test.qp_eq ${LDLT_TEST_LIBS})

add_executable(.test.precond.ruiz src/precond_ruiz.cpp)
target_link_libraries(.test.precond.ruiz ${LDLT_TEST_LIBS})
