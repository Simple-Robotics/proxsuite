add_subdirectory(cpp)

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif(BUILD_PYTHON_INTERFACE)

if(TEST_JULIA_INTERFACE)
  find_package(Julia REQUIRED)

  julia_check_package(PyCall)
  if(NOT Julia_PyCall_found)
    message(
      FATAL_ERROR
        "PyCall not installed. Please use import Pkg; Pkg.add(\"PyCall\")")
  endif()
  add_subdirectory(julia)
endif()
