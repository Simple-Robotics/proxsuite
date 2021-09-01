include_directories(include)

target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)

set(benchlibs inria_ldlt backward_cpp_main)

add_executable(.bench.factorize src/factorize.cpp)
target_link_libraries(.bench.factorize ${benchlibs} CONAN_PKG::benchmark)

add_executable(.bench.qp_eq src/qp_eq.cpp)
target_link_libraries(.bench.qp_eq ${benchlibs} CONAN_PKG::benchmark)

add_executable(.bench.osqp src/osqp.cpp)
target_link_libraries(.bench.osqp ${benchlibs} CONAN_PKG::benchmark)
