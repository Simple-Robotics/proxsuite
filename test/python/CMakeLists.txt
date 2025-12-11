function(proxsuite_python_add_test name)
  add_test(
    NAME proxsuite-test-python.${name}
    COMMAND $<TARGET_FILE:Python::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
  )
  set_tests_properties(
    proxsuite-test-python.${name}
    PROPERTIES
      ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:proxsuite_pywrap>/.."
      LABELS python
  )
endfunction()

proxsuite_python_add_test(test_cvxpy)
proxsuite_python_add_test(test_dense_qp_solve)
proxsuite_python_add_test(test_dense_qp_wrapper)

if(BUILD_WITH_OPENMP_SUPPORT)
  proxsuite_python_add_test(test_parallel_qp_solve)
endif()

if(PROXSUITE_BUILD_WITH_SERIALIZATION)
  proxsuite_python_add_test(test_serialization)
endif()

proxsuite_python_add_test(test_sparse_qp_solve)
proxsuite_python_add_test(test_sparse_qp_wrapper)
