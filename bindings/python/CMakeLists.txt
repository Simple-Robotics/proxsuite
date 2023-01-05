include(../../cmake-module/python.cmake)

findpython(REQUIRED Development.Module)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/pybind11)
add_subdirectory(external/pybind11
                 ${CMAKE_CURRENT_BINARY_DIR}/external/pybind11)

if(IS_ABSOLUTE ${PYTHON_SITELIB})
  set(${PYWRAP}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
else()
  set(${PYWRAP}_INSTALL_DIR
      ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB}/${PROJECT_NAME})
endif()

add_custom_target(python)

# Collect files
file(GLOB_RECURSE PYWRAP_HEADERS ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)
file(GLOB_RECURSE PYWRAP_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# Add simd feature detectors for current intel CPU
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  python3_add_library(instructionset MODULE helpers/instruction-set.cpp)
  add_dependencies(python instructionset)
  target_link_libraries(instructionset PRIVATE proxsuite pybind11::module)
  set_target_properties(
    instructionset
    PROPERTIES OUTPUT_NAME instructionset
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
               LIBRARY_OUTPUT_DIRECTORY_RELEASE
               "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
               LIBRARY_OUTPUT_DIRECTORY_DEBUG
               "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}")
  if(UNIX AND NOT APPLE)
    set_target_properties(instructionset PROPERTIES INSTALL_RPATH
                                                    "\$ORIGIN/../../..")
  endif()
  install(
    TARGETS instructionset
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION ${${PYWRAP}_INSTALL_DIR})
endif()

function(CREATE_PYTHON_TARGET target_name COMPILE_OPTIONS dependencies)
  python3_add_library(${target_name} MODULE ${PYWRAP_SOURCES} ${PYWRAP_HEADERS})
  add_dependencies(python ${target_name})

  target_link_libraries(${target_name} PUBLIC ${dependencies} pybind11::module)
  target_compile_options(${target_name} PRIVATE ${COMPILE_OPTIONS})
  target_link_libraries(${target_name} PRIVATE proxsuite pybind11::module)
  target_compile_definitions(${target_name}
                             PRIVATE PYTHON_MODULE_NAME=${target_name})
  target_include_directories(
    ${target_name} SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/cereal/include)
  set_target_properties(
    ${target_name}
    PROPERTIES OUTPUT_NAME ${target_name}
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
               LIBRARY_OUTPUT_DIRECTORY_RELEASE
               "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
               LIBRARY_OUTPUT_DIRECTORY_DEBUG
               "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}")

  if(UNIX AND NOT APPLE)
    set_target_properties(${target_name} PROPERTIES INSTALL_RPATH
                                                    "\$ORIGIN/../../..")
  endif()

  install(TARGETS ${target_name} DESTINATION ${${PYWRAP}_INSTALL_DIR})
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
  set(AVX_COMPILE_OPTION "/arch:AVX")
  set(AVX2_COMPILE_OPTION "/arch:AVX2")
  set(FMA_COMPILE_OPTION "/fp:fast,")
  set(AVX512_COMPILE_OPTION "/arch:AVX512")
else()
  set(AVX_COMPILE_OPTION "-mavx")
  set(AVX2_COMPILE_OPTION "-mavx2")
  set(FMA_COMPILE_OPTION "-mfma")
  set(AVX512_COMPILE_OPTION "-mavx512f")
endif()

create_python_target(proxsuite_pywrap "" proxsuite)
if(BUILD_WITH_VECTORIZATION_SUPPORT AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES
                                        "(x86)|(X86)|(amd64)|(AMD64)")
  if(BUILD_BINDINGS_WITH_AVX2_SUPPORT)
    create_python_target(
      proxsuite_pywrap_avx2 "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}"
      proxsuite-vectorized)
  endif(BUILD_BINDINGS_WITH_AVX2_SUPPORT)
  if(BUILD_BINDINGS_WITH_AVX512_SUPPORT)
    create_python_target(
      proxsuite_pywrap_avx512 "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}"
      proxsuite-vectorized)
  endif(BUILD_BINDINGS_WITH_AVX512_SUPPORT)
endif()
add_header_group(PYWRAP_HEADERS)
add_source_group(PYWRAP_SOURCES)

# --- INSTALL SCRIPTS
set(PYTHON_FILES __init__.py)

foreach(python ${PYTHON_FILES})
  python_build(${PROJECT_NAME} ${python})
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/proxsuite/${python}"
          DESTINATION ${${PYWRAP}_INSTALL_DIR})
endforeach(python)
add_dependencies(python ${compile_pyc})
