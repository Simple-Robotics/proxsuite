macro(run_conan)
  option(CONAN_QUIET "Suppress Conan output" OFF)
  # Download automatically, you can also just copy the conan.cmake file
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(
      STATUS
        "Downloading conan.cmake from https://github.com/conan-io/cmake-conan"
    )
    file(DOWNLOAD
         "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
         "${CMAKE_BINARY_DIR}/conan.cmake"
    )
  endif()

  include(${CMAKE_BINARY_DIR}/conan.cmake)

  conan_check(REQUIRED)
  conan_add_remote(
    COMMAND
    ${CONAN_CMD}
    NAME
    conan-center
    URL
    https://conan.bintray.com
  )
  conan_add_remote(
    COMMAND
    ${CONAN_CMD}
    NAME
    bincrafters
    URL
    https://api.bintray.com/conan/bincrafters/public-conan
  )

  option(CONAN_QUIET "Suppress Conan output" OFF)
  if(CONAN_QUIET)
    set(OUTPUT_QUIET OUTPUT_QUIET)
  endif()
  conan_cmake_run(
    ${OUTPUT_QUIET}
    REQUIRES
    ${CONAN_REQUIRES}
    OPTIONS
    ${CONAN_OPTIONS}
    BASIC_SETUP
    CMAKE_TARGETS # individual targets to link to
    BUILD
    missing
  )
endmacro()
