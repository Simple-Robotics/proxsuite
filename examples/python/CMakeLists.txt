file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_EXAMPLES *.py)

foreach(EXAMPLE_FILE ${${PROJECT_NAME}_PYTHON_EXAMPLES})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
  string(REGEX REPLACE "${PROJECT_SOURCE_DIR}" "" EXAMPLE_FILE ${EXAMPLE_FILE})
  add_python_unit_test("example-py-${EXAMPLE_NAME}" "${EXAMPLE_FILE}"
                       "bindings/python")
endforeach(EXAMPLE_FILE ${${PROJECT_NAME}_PYTHON_EXAMPLES})
