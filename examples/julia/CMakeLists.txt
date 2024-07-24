file(GLOB_RECURSE ${PROJECT_NAME}_JULIA_EXAMPLES *.jl)

foreach(EXAMPLE ${${PROJECT_NAME}_JULIA_EXAMPLES})
  string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/examples/julia/" "" EXAMPLE
                       ${EXAMPLE})
  add_julia_unit_test("${PROJECT_NAME}-example-jl-${EXAMPLE}"
                      "examples/julia/${EXAMPLE}")
endforeach()
