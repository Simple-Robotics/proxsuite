include(../cmake/doctest.cmake)

include_directories(include)

target_link_libraries(
  inria_ldlt INTERFACE CONAN_PKG::fmt CONAN_PKG::eigen CONAN_PKG::mpfr
)
target_include_directories(
  inria_ldlt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(testlibs inria_ldlt backward_cpp_main doctest_main)

add_executable(.test.factorize.roundtrip src/ldlt_roundtrip.cpp)
target_link_libraries(.test.factorize.roundtrip ${testlibs})
