include(../cmake-external/doctest.cmake)
find_package(Matio REQUIRED)

add_library(${PROJECT_NAME}-doctest STATIC doctest/doctest.cpp)
target_include_directories(${PROJECT_NAME}-doctest PUBLIC ./doctest)
add_library(cnpy OBJECT src/cnpy.cpp)
target_link_libraries(cnpy Eigen3::Eigen)
target_include_directories(cnpy PUBLIC ./include)

file(GLOB_RECURSE TEST_HEADERS ./include/*.hpp)
ADD_HEADER_GROUP(TEST_HEADERS)

add_library(
  proxsuite-test-util
  STATIC
  ./src/util_f64.cpp
  ./src/util_f32.cpp
  ${TEST_HEADERS}
)
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
  target_link_libraries(
    ${target_name}
    PUBLIC proxsuite ${PROJECT_NAME}-doctest proxsuite-test-util
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE PROBLEM_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  add_dependencies(build_tests ${target_name})
endmacro()

proxsuite_test(proxqp-dense_ruiz_equilibration src/proxqp/dense_ruiz_equilibration.cpp)
proxsuite_test(proxqp-dense_qp_eq src/proxqp/dense_qp_eq.cpp)
proxsuite_test(proxqp-dense_qp_with_eq_and_in src/proxqp/dense_qp_with_eq_and_in.cpp)
proxsuite_test(proxqp-dense_qp_unconstrained src/proxqp/dense_unconstrained_qp.cpp)
proxsuite_test(proxqp-dense_backward src/proxqp/dense_backward.cpp)
proxsuite_test(proxqp-dense_qp_wrapper src/proxqp/dense_qp_wrapper.cpp)
proxsuite_test(proxqp-dense_qp_solve src/proxqp/dense_qp_solve.cpp)
proxsuite_test(proxqp-sparse_ruiz_equilibration src/proxqp/sparse_ruiz_equilibration.cpp)
proxsuite_test(proxqp-sparse_qp src/proxqp/sparse_qp.cpp)
proxsuite_test(proxqp-sparse_qp_wrapper src/proxqp/sparse_qp_wrapper.cpp)
proxsuite_test(proxqp-sparse_qp_solve src/proxqp/sparse_qp_solve.cpp)
proxsuite_test(proxqp-sparse_factorization src/proxqp/sparse_factorization.cpp)
proxsuite_test(proxqp-cvxpy src/proxqp/cvxpy.cpp)

proxsuite_test(osqp-cvxpy src/osqp/cvxpy.cpp)
proxsuite_test(osqp-dense_qp_eq src/osqp/dense_qp_eq.cpp)
proxsuite_test(osqp-dense_qp_solve src/osqp/dense_qp_solve.cpp)
proxsuite_test(osqp-dense_qp_with_eq_and_in src/osqp/dense_qp_with_eq_and_in.cpp)
proxsuite_test(osqp-dense_qp_wrapper src/osqp/dense_qp_wrapper.cpp)
proxsuite_test(osqp-dense_ruiz_equilibration src/osqp/dense_ruiz_equilibration.cpp)
proxsuite_test(osqp-dense_qp_unconstrained src/osqp/dense_unconstrained_qp.cpp)

if(BUILD_WITH_OPENMP_SUPPORT)
  proxsuite_test(proxqp-parallel src/proxqp/parallel_qp_solve.cpp)
  target_link_libraries(
    ${PROJECT_NAME}-test-cpp-proxqp-parallel
    PRIVATE OpenMP::OpenMP_CXX
  )
endif()

# Test serialization
macro(ADD_TEST_CFLAGS target flag)
  set_property(TARGET ${target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
endmacro()

make_directory("${CMAKE_CURRENT_BINARY_DIR}/serialization-data")
proxsuite_test(proxqp-serialization src/proxqp/serialization.cpp)
add_test_cflags(
  ${PROJECT_NAME}-test-cpp-proxqp-serialization
  "-DTEST_SERIALIZATION_FOLDER=\\\\\"${CMAKE_CURRENT_BINARY_DIR}/serialization-data\\\\\""
)
if(cereal_FOUND)
  target_link_libraries(
    ${PROJECT_NAME}-test-cpp-proxqp-serialization
    PRIVATE cereal::cereal
  )
else()
  target_include_directories(
    ${PROJECT_NAME}-test-cpp-proxqp-serialization
    SYSTEM
    PRIVATE ${PROJECT_SOURCE_DIR}/external/cereal/include
  )
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
  proxsuite_test(proxqp-dense_maros_meszaros src/proxqp/dense_maros_meszaros.cpp)
  proxsuite_test(proxqp-sparse_maros_meszaros src/proxqp/sparse_maros_meszaros.cpp)

  proxsuite_test(osqp-dense_maros_meszaros src/osqp/dense_maros_meszaros.cpp)
endif()

if(BUILD_PYTHON_INTERFACE)
  file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_UNITTEST *.py)

  if(
    (CMAKE_BUILD_TYPE STREQUAL "Debug" AND MSVC)
    OR NOT BUILD_WITH_OPENMP_SUPPORT
  )
    list_filter(
      "${${PROJECT_NAME}_PYTHON_UNITTEST}"
      "parallel_qp_solve"
      ${PROJECT_NAME}_PYTHON_UNITTEST
    )
  endif()

  foreach(TEST_FILE ${${PROJECT_NAME}_PYTHON_UNITTEST})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    string(REPLACE "${PROJECT_SOURCE_DIR}/" "" TEST_FILE_REL ${TEST_FILE})
    string(REGEX REPLACE "^test/src/" "" TEST_SUBPATH ${TEST_FILE_REL})
    get_filename_component(TEST_DIR ${TEST_SUBPATH} DIRECTORY)
    if(TEST_DIR STREQUAL "")
      set(PYTHON_TEST_NAME "${PROJECT_NAME}-test-py-${TEST_NAME}")
    else()
      string(REPLACE "/" "-" TEST_DIR_CLEAN ${TEST_DIR})
      set(
        PYTHON_TEST_NAME
        "${PROJECT_NAME}-test-py-${TEST_DIR_CLEAN}-${TEST_NAME}"
      )
    endif()
    ADD_PYTHON_UNIT_TEST(
      "${PYTHON_TEST_NAME}"
      "${TEST_FILE_REL}"
      "bindings/python"
    )
  endforeach()
endif(BUILD_PYTHON_INTERFACE)
