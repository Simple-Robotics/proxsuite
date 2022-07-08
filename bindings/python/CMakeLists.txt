include(../../cmake-module/python.cmake)

findpython(REQUIRED)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/ pybind11) 
add_subdirectory(external/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/external/pybind11)

set(PYWRAP proxsuite_pywrap)
add_library(proxsuite_pywrap MODULE src/expose-all.cpp)

target_link_libraries(proxsuite_pywrap PUBLIC proxsuite pybind11::module)

if(IS_ABSOLUTE ${PYTHON_SITELIB})
	set(${PYWRAP}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
else()
	set(${PYWRAP}_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB}/${PROJECT_NAME})
endif()

set_target_properties(${PYWRAP}
	PROPERTIES
	PREFIX ""
	SUFFIX ${PYTHON_EXT_SUFFIX}
	OUTPUT_NAME proxsuite_pywrap
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
)   

if(UNIX AND NOT APPLE)
	set_target_properties(proxsuite_pywrap PROPERTIES INSTALL_RPATH "\$ORIGIN/../../..")
endif()

install(
	TARGETS proxsuite_pywrap
	EXPORT ${TARGETS_EXPORT_NAME}
	DESTINATION ${${PYWRAP}_INSTALL_DIR}
)