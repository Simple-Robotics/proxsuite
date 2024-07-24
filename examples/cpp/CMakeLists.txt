#
# Copyright (c) 2022 INRIA
#

add_custom_target(${PROJECT_NAME}-example-cpp)

function(ADD_PROXSUITE_CPP_EXAMPLE EXAMPLE)
  get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
  set(EXAMPLE_TARGET "${PROJECT_NAME}-example-cpp-${EXAMPLE_NAME}")
  add_unit_test(${EXAMPLE_TARGET} "${EXAMPLE}")
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE proxsuite-test-util)

  add_dependencies(${PROJECT_NAME}-example-cpp ${EXAMPLE_TARGET})
endfunction()

file(GLOB_RECURSE ${PROJECT_NAME}_CPP_EXAMPLES *.cpp)

foreach(EXAMPLE ${${PROJECT_NAME}_CPP_EXAMPLES})
  add_proxsuite_cpp_example(${EXAMPLE})
endforeach()
