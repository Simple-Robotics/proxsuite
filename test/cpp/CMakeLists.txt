add_library(proxsuite-test-util STATIC util_f64.cpp util_f32.cpp)
target_link_libraries(proxsuite-test-util PUBLIC proxsuite)

if(BUILD_WITH_VECTORIZATION_SUPPORT)
  add_library(proxsuite-test-util-vectorized STATIC util_f64.cpp util_f32.cpp)
  target_include_directories(proxsuite-test-util-vectorized PRIVATE include)
  target_link_libraries(proxsuite-test-util-vectorized PUBLIC proxsuite-vectorized)
endif()

function(proxsuite_add_test name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs COMPILE_DEFINITIONS LINK_LIBRARIES)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(proxsuite-test-cpp-${name} ${name}.cpp)
  target_link_libraries(
    proxsuite-test-cpp-${name}
    PRIVATE proxsuite-test-util Catch2::Catch2WithMain ${arg_LINK_LIBRARIES}
  )
  target_compile_definitions(proxsuite-test-cpp-${name} PRIVATE ${arg_COMPILE_DEFINITIONS})
  add_test(NAME proxsuite-test-cpp.${name} COMMAND proxsuite-test-cpp-${name})

  if(BUILD_WITH_VECTORIZATION_SUPPORT)
    add_executable(proxsuite-vectorized-test-cpp-${name} ${name}.cpp)
    target_link_libraries(
      proxsuite-vectorized-test-cpp-${name}
      PRIVATE proxsuite-test-util-vectorized Catch2::Catch2WithMain ${arg_LINK_LIBRARIES}
    )
    target_compile_definitions(
      proxsuite-vectorized-test-cpp-${name}
      PRIVATE ${arg_COMPILE_DEFINITIONS}
    )
    add_test(
      NAME proxsuite-test-cpp.vectorized.${name}
      COMMAND proxsuite-vectorized-test-cpp-${name}
    )
  endif()
endfunction()

proxsuite_add_test(uint128)
proxsuite_add_test(cvxpy)
proxsuite_add_test(dense_backward)
proxsuite_add_test(dense_qp_eq)
proxsuite_add_test(dense_qp_solve)
proxsuite_add_test(dense_qp_with_eq_and_in)
proxsuite_add_test(dense_qp_wrapper)
proxsuite_add_test(dense_ruiz_equilibration)
proxsuite_add_test(dense_unconstrained_qp)
proxsuite_add_test(sparse_factorization)
proxsuite_add_test(sparse_qp_solve)
proxsuite_add_test(sparse_qp_wrapper)
proxsuite_add_test(sparse_qp)
proxsuite_add_test(sparse_ruiz_equilibration)

if(BUILD_WITH_OPENMP_SUPPORT)
  proxsuite_add_test(parallel_qp_solve)
endif()

if(BUILD_WITH_SERIALIZATION)
  # We have to create the folder for the output data, tests don't do that.
  make_directory(${CMAKE_BINARY_DIR}/serialization-data)
  proxsuite_add_test(serialization COMPILE_DEFINITIONS TEST_SERIALIZATION_FOLDER="${CMAKE_BINARY_DIR}/serialization-data")
endif()

if(BUILD_MAROS_MESZAROS_TESTS)
  proxsuite_add_test(dense_maros_meszaros COMPILE_DEFINITIONS MAROS_MESZAROS_DIR="${MAROS_MESZAROS_DIR}" LINK_LIBRARIES matio::matio)
endif()
