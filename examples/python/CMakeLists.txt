add_custom_target(proxsuite-examples-python)

function(proxsuite_add_python_example name)
  add_custom_target(
    proxsuite-example-py-${name}
    COMMAND $<TARGET_FILE:Python::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
    DEPENDS proxsuite_pywrap
  )
  if(BUILD_TESTING)
    add_test(
      NAME proxsuite-example-python.${name}
      COMMAND $<TARGET_FILE:Python::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
    )
    set_tests_properties(
      proxsuite-example-python.${name}
      PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:proxsuite_pywrap>/.." LABELS python
    )
  endif()
  add_dependencies(proxsuite-examples-python proxsuite-example-py-${name})
endfunction()

proxsuite_add_python_example(estimate_nonconvex_eigenvalue)
proxsuite_add_python_example(init_dense_qp_with_box)
proxsuite_add_python_example(init_dense_qp_with_other_options)
proxsuite_add_python_example(init_dense_qp_with_timings)
proxsuite_add_python_example(init_dense_qp)
proxsuite_add_python_example(init_with_default_options)
proxsuite_add_python_example(initializing_with_none_without_api)
proxsuite_add_python_example(initializing_with_none)
proxsuite_add_python_example(loading_dense_qp_with_box_ineq)
proxsuite_add_python_example(loading_dense_qp_with_different_backend_choice)
proxsuite_add_python_example(loading_dense_qp)
proxsuite_add_python_example(loading_sparse_qp)
proxsuite_add_python_example(overview-simple)
proxsuite_add_python_example(qplayer_sudoku)
proxsuite_add_python_example(solve_dense_lp)
proxsuite_add_python_example(solve_dense_qp_with_setting)
proxsuite_add_python_example(solve_dense_qp)
proxsuite_add_python_example(solve_without_api_and_option)
proxsuite_add_python_example(solve_without_api)
proxsuite_add_python_example(update_dense_qp_ws_previous_result)
proxsuite_add_python_example(update_dense_qp)
proxsuite_add_python_example(update_sparse_qp)
