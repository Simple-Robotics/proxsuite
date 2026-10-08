find_program(Julia_EXECUTABLE julia REQUIRED)
message(STATUS "Found Julia executable: ${Julia_EXECUTABLE}")

add_test(
  NAME proxsuite-example-jl-overview-simple
  COMMAND ${Julia_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/overview-simple.jl
)
set_property(
  TEST proxsuite-example-jl-overview-simple
  PROPERTY
    ENVIRONMENT
      "PYTHON=$<TARGET_FILE:Python::Interpreter>"
      "PYTHONPATH=$<TARGET_FILE_DIR:proxsuite_pywrap>/.."
)
