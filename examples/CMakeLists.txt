add_subdirectory(cpp)

IF(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
ENDIF(BUILD_PYTHON_INTERFACE)

if(Julia_FOUND)
  JULIA_CHECK_PACKAGE(PyCall)
  if(Julia_PyCall_found)
    add_subdirectory(julia)
  endif()
endif()
