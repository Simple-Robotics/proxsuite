option(ENABLE_IPO
       "Enable Iterprocedural Optimization, aka Link Time Optimization (LTO)"
       OFF
)
option(ARCH_NATIVE "Build with -march=native" OFF)
option(USE_LIBCXX "Use the libc++ STL" OFF)

option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
if(ENABLE_BUILD_WITH_TIME_TRACE AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  add_compile_options("-ftime-trace")
endif()

if(USE_LLD)
  add_link_options("-fuse-ld=lld")
elseif(USE_GOLD)
  add_link_options("-fuse-ld=gold")
endif()

if(USE_LIBCXX)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    add_compile_options("-stdlib=libc++")
    add_link_options("-stdlib=libc++ -lc++abi")
  else()
    message(SEND_ERROR "libc++ only available with clang")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Choose the type of build." FORCE
  )
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
             "Debug"
             "Release"
             "MinSizeRel"
             "RelWithDebInfo"
  )
endif()

if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(
    RESULT
    result
    OUTPUT
    output
  )
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(SEND_ERROR "IPO is not supported: ${output}")
  endif()
endif()

include(CheckCXXCompilerFlag)
if(ARCH_NATIVE)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options("-march=native")
  endif()
endif()
