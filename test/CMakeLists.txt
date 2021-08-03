include(../cmake/doctest.cmake)

include_directories(include)

# Conan dependencies
set(CONAN_REQUIRES
    # MIT License
    fmt/8.0.0
    backward-cpp/1.5
    #
    # Mozilla Public License 2.0
    eigen/3.3.9
    #
    # LGPL v3
    mpfr/4.1.0
)
set(CONAN_OPTIONS)
run_conan()

target_link_libraries(
  inria_ldlt INTERFACE CONAN_PKG::fmt CONAN_PKG::eigen CONAN_PKG::mpfr
)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(testlibs inria_ldlt backward_cpp_main doctest_main)

add_executable(ldlt_roundtrip src/ldlt_roundtrip.cpp)
target_link_libraries(ldlt_roundtrip ${testlibs})
