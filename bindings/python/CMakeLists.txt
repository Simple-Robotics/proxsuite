include(../../cmake-module/python.cmake)

findpython(REQUIRED)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/ pybind11)
add_subdirectory(external/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/external/pybind11)

if(IS_ABSOLUTE ${PYTHON_SITELIB})
	set(${PYWRAP}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
else()
	set(${PYWRAP}_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB}/${PROJECT_NAME})
endif()

# Collect files
file(GLOB_RECURSE PYWRAP_HEADERS
  ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
)
add_header_group(PYWRAP_HEADERS)

file(GLOB_RECURSE PYWRAP_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_source_group(PYWRAP_SOURCES)

macro(CREATE_PYTHON_TARGET target_name COMPILE_OPTIONS dependencies)
	add_library(${target_name} MODULE ${PYWRAP_SOURCES} ${PYWRAP_HEADERS})

	target_link_libraries(${target_name} PUBLIC ${dependencies} pybind11::module)
	target_compile_options(${target_name} PRIVATE ${COMPILE_OPTIONS})
	target_compile_definitions(${target_name} PRIVATE PYTHON_MODULE_NAME=${target_name})

	set_target_properties(${target_name}
		PROPERTIES
		PREFIX ""
		SUFFIX ${PYTHON_EXT_SUFFIX}
		OUTPUT_NAME ${target_name}
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
	)

	if(UNIX AND NOT APPLE)
		set_target_properties(${target_name} PROPERTIES INSTALL_RPATH "\$ORIGIN/../../..")
	endif()

	install(
		TARGETS ${target_name}
		EXPORT ${TARGETS_EXPORT_NAME}
		DESTINATION ${${PYWRAP}_INSTALL_DIR}
	)
endmacro()

CREATE_PYTHON_TARGET(proxsuite_pywrap "" proxsuite-not-vectorized)
if(BUILD_WITH_VECTORIZATION_SUPPORT)
  CREATE_PYTHON_TARGET(proxsuite_pywrap_avx2 "-mavx2" proxsuite-vectorized)
  CREATE_PYTHON_TARGET(proxsuite_pywrap_avx512 "-mavx512f" proxsuite-vectorized)
endif()
