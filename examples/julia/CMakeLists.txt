find_program(Julia_EXECUTABLE julia REQUIRED)
message(STATUS "Found Julia executable: ${Julia_EXECUTABLE}")

add_test(
  NAME proxsuite-example-jl-overview-simple
  COMMAND ${Julia_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/overview-simple.jl
)
set_tests_properties(
  proxsuite-example-jl-overview-simple
  PROPERTIES
    ENVIRONMENT PYTHON=$<TARGET_FILE:Python::Interpreter>
    ENVIRONMENT PYTHONPATH=$<TARGET_FILE_DIR:proxsuite_pywrap>/..
    LABELS julia
    DEPENDS proxsuite_pywrap
)
