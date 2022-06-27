
find_package(Python3 COMPONENTS Interpreter Development)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11) 
add_subdirectory(../../external/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
add_library(proxsuite_pywrap MODULE src/expose-all.cpp)
target_compile_definitions(
	proxsuite_pywrap INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN
)

execute_process(
	COMMAND
		"${Python3_EXECUTABLE}" "-c"
		"from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"
	OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
)

string(STRIP ${PYTHON_EXT_SUFFIX} PYTHON_EXT_SUFFIX)
message(STATUS "PYTHON_EXT_SUFFIX: ${PYTHON_EXT_SUFFIX}")
target_link_libraries(proxsuite_pywrap PUBLIC proxsuite pybind11::module)
set_target_properties(
	proxsuite_pywrap
	PROPERTIES PREFIX ""
						 SUFFIX "${PYTHON_EXT_SUFFIX}"
)