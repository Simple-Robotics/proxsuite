find_package(Doxygen)

if(DOXYGEN_FOUND)
  get_target_property(
    PROXSUITE_PUBLIC_HEADER_DIR prox-suite INTERFACE_INCLUDE_DIRECTORIES
  )
  file(GLOB_RECURSE PROXSUITE_PUBLIC_HEADERS
       ${PROXSUITE_PUBLIC_HEADER_DIR}/*.hpp
  )
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/../docs/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  add_custom_target(
    doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    DEPENDS ${PROXSUITE_PUBLIC_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating docs"
    MAIN_DEPENDENCY Doxyfile
    VERBATIM
  )
else(DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif(DOXYGEN_FOUND)
