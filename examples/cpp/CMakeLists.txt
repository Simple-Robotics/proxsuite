#
# Copyright (c) 2022 INRIA
#

add_custom_target(${PROJECT_NAME}-example-cpp)

file(GLOB_RECURSE ${PROJECT_NAME}_CPP_EXAMPLES *.cpp)
foreach(EXAMPLE_FILE ${${PROJECT_NAME}_CPP_EXAMPLES})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
  string(
    REGEX REPLACE
    "${PROJECT_SOURCE_DIR}/"
    ""
    EXAMPLE_FILE_REL
    ${EXAMPLE_FILE}
  )

  string(REGEX REPLACE "^examples/cpp/" "" EXAMPLE_SUBPATH ${EXAMPLE_FILE_REL})
  get_filename_component(EXAMPLE_DIR ${EXAMPLE_SUBPATH} DIRECTORY)

  if(EXAMPLE_DIR STREQUAL "")
    set(EXAMPLE_TARGET "${PROJECT_NAME}-example-cpp-${EXAMPLE_NAME}")
  else()
    string(REPLACE "/" "-" EXAMPLE_DIR_CLEAN ${EXAMPLE_DIR})
    set(
      EXAMPLE_TARGET
      "${PROJECT_NAME}-example-cpp-${EXAMPLE_DIR_CLEAN}-${EXAMPLE_NAME}"
    )
  endif()

  ADD_UNIT_TEST(${EXAMPLE_TARGET} "${EXAMPLE_FILE}")
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE proxsuite-test-util)
  add_dependencies(${PROJECT_NAME}-example-cpp ${EXAMPLE_TARGET})
endforeach()
