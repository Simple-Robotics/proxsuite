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

add_library(nanobench_main STATIC src/nanobench_main.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)

set(benchlibs inria_ldlt backward_cpp_main nanobench_main)
