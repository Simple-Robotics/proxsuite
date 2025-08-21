file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_EXAMPLES *.py)
foreach(EXAMPLE_FILE ${${PROJECT_NAME}_PYTHON_EXAMPLES})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)

  if(EXAMPLE_FILE MATCHES ".*/calibration/.*")
    continue()
  endif()

  string(
    REGEX REPLACE
    "${PROJECT_SOURCE_DIR}/"
    ""
    EXAMPLE_FILE_REL
    ${EXAMPLE_FILE}
  )

  string(
    REGEX REPLACE
    "^examples/python/"
    ""
    EXAMPLE_SUBPATH
    ${EXAMPLE_FILE_REL}
  )
  get_filename_component(EXAMPLE_DIR ${EXAMPLE_SUBPATH} DIRECTORY)

  if(EXAMPLE_DIR STREQUAL "")
    set(EXAMPLE_TARGET "${PROJECT_NAME}-example-py-${EXAMPLE_NAME}")
  else()
    string(REPLACE "/" "-" EXAMPLE_DIR_CLEAN ${EXAMPLE_DIR})
    set(
      EXAMPLE_TARGET
      "${PROJECT_NAME}-example-py-${EXAMPLE_DIR_CLEAN}-${EXAMPLE_NAME}"
    )
  endif()

  ADD_PYTHON_UNIT_TEST(
    "${EXAMPLE_TARGET}"
    "${EXAMPLE_FILE_REL}"
    "bindings/python"
  )
endforeach()
