set(
  proxsuite_pywrap_HEADERS
  src/algorithms.hpp
  src/expose-backward.hpp
  src/expose-helpers.hpp
  src/expose-model.hpp
  src/expose-qpobject.hpp
  src/expose-qpvector.hpp
  src/expose-results.hpp
  src/expose-settings.hpp
  src/expose-solve.hpp
  src/expose-workspace.hpp
  src/optional-eigen-fix.hpp
)
if(BUILD_WITH_OPENMP_SUPPORT)
  list(APPEND proxsuite_pywrap_HEADERS src/expose-parallel.hpp)
endif()

set(proxsuite_pywrap_SOURCES src/expose-all.cpp)

function(create_python_target target_name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs COMPILE_OPTIONS LINK_LIBRARIES)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  nanobind_add_module(${target_name} NB_STATIC LTO NB_SUPPRESS_WARNINGS ${proxsuite_pywrap_SOURCES} ${proxsuite_pywrap_HEADERS})
  jrl_check_python_module_name(${target_name})

  target_link_libraries(${target_name} PRIVATE ${arg_LINK_LIBRARIES})
  target_compile_options(${target_name} PRIVATE ${arg_COMPILE_OPTIONS})
  target_compile_definitions(${target_name} PRIVATE PYTHON_MODULE_NAME=${target_name})
  jrl_target_set_output_directory(${target_name} OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/${PROJECT_NAME})
endfunction()

# copy __init_.py and torch files into the build directory (so we can import proxsuite)
file(COPY proxsuite DESTINATION ${CMAKE_BINARY_DIR}/lib/site-packages)

# The main python binding target
create_python_target(proxsuite_pywrap LINK_LIBRARIES proxsuite)

# Vectorized python binding targets
if(
  BUILD_WITH_VECTORIZATION_SUPPORT
  AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)"
)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(AVX_COMPILE_OPTION "/arch:AVX")
    set(AVX2_COMPILE_OPTION "/arch:AVX2")
    set(FMA_COMPILE_OPTION "/fp:fast")
    set(AVX512_COMPILE_OPTION "/arch:AVX512")
  else()
    set(AVX_COMPILE_OPTION "-mavx")
    set(AVX2_COMPILE_OPTION "-mavx2")
    set(FMA_COMPILE_OPTION "-mfma")
    set(AVX512_COMPILE_OPTION "-mavx512f")
  endif()

  if(BUILD_BINDINGS_WITH_AVX2_SUPPORT)
    message(STATUS "[proxsuite] Building python bindings with AVX2 support")
    create_python_target(
      proxsuite_pywrap_avx2
      COMPILE_OPTIONS "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}"
      LINK_LIBRARIES proxsuite-vectorized
    )
  endif()

  if(BUILD_BINDINGS_WITH_AVX512_SUPPORT)
    message(STATUS "[proxsuite] Building python bindings with AVX512 support")
    create_python_target(
      proxsuite_pywrap_avx512
      COMPILE_OPTIONS "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}"
      LINK_LIBRARIES proxsuite-vectorized
    )
  endif()
endif()

# Add simd feature detectors for current intel CPU
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  nanobind_add_module(instructionset NB_STATIC LTO NB_SUPPRESS_WARNINGS helpers/instruction-set.cpp)
  target_link_libraries(instructionset PRIVATE proxsuite)
  jrl_target_set_output_directory(
    instructionset
    OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib/site-packages/${PROJECT_NAME}
  )
  if(GENERATE_PYTHON_STUBS)
    nanobind_add_stub(
      instructionset_stub
      MODULE instructionset
      OUTPUT ${CMAKE_BINARY_DIR}/lib/site-packages/${PROJECT_NAME}/instructionset.pyi
      PYTHON_PATH $<TARGET_FILE_DIR:instructionset>
      DEPENDS instructionset
    )
  endif()
endif()

if(GENERATE_PYTHON_STUBS)
  set(simd_lib "")
  if(TARGET proxsuite_pywrap_avx2)
    set(simd_lib proxsuite_pywrap_avx2)
  endif()
  if(TARGET proxsuite_pywrap_avx512)
    set(simd_lib proxsuite_pywrap_avx512)
  endif()

  nanobind_add_stub(
    proxsuite_pywrap_stub
    MODULE proxsuite
    VERBOSE
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/site-packages/${PROJECT_NAME}
    OUTPUT
      __init__.pyi
      helpers.pyi
      proxqp/__init__.pyi
      proxqp/dense.pyi
      proxqp/sparse.pyi
    RECURSIVE
    PYTHON_PATH ${CMAKE_BINARY_DIR}/lib/site-packages
    DEPENDS proxsuite_pywrap ${simd_lib}
  )
endif()

jrl_python_compile_all(
  DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages
)

# Install the whole python package directory
jrl_python_compute_install_dir(python_install_dir)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/${PROJECT_NAME}
  DESTINATION ${python_install_dir}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyc"
  PATTERN "*.pyi"
  PATTERN "*.typed"
  PATTERN "*.so"
  PATTERN "*.pyd"
)
