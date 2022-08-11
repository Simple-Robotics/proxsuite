add_custom_target(example-py)

file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_EXAMPLES *.py)

foreach(EXAMPLE ${${PROJECT_NAME}_PYTHON_EXAMPLES})
  string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/examples/python/" "" EXAMPLE
                       ${EXAMPLE})
  add_python_unit_test("example-py-${EXAMPLE}" "examples/python/${EXAMPLE}"
                       "bindings/python")
endforeach(EXAMPLE ${${PROJECT_NAME}_PYTHON_EXAMPLES})
