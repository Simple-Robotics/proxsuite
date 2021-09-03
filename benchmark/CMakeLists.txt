include_directories(include)

target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)

set(LDLT_BENCH_LIBS inria_ldlt backward_cpp_main libcnpy)

add_executable(.bench.factorize src/factorize.cpp)
target_link_libraries(.bench.factorize ${LDLT_BENCH_LIBS} CONAN_PKG::benchmark)

add_executable(.bench.qp_eq src/qp_eq.cpp)
target_link_libraries(.bench.qp_eq ${LDLT_BENCH_LIBS} CONAN_PKG::benchmark)

add_executable(.bench.osqp src/osqp.cpp)
target_link_libraries(.bench.osqp ${LDLT_BENCH_LIBS} CONAN_PKG::benchmark)

add_executable(.bench.osqp.py src/osqp_py.cpp)
target_link_libraries(.bench.osqp.py ${LDLT_BENCH_LIBS} osqpstatic)
target_compile_definitions(
  .bench.osqp.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${INRIA_LDLT_QP_PYTHON_PATH}"
)
add_dependencies(.bench.osqp.py .test.qp_eq.py.generated_problems)

add_executable(.bench.qpalm.py src/qpalm_py.cpp)
target_link_libraries(.bench.qpalm.py ${LDLT_BENCH_LIBS} qpalm)
target_compile_definitions(
  .bench.qpalm.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${INRIA_LDLT_QP_PYTHON_PATH}"
)
add_dependencies(.bench.qpalm.py .test.qp_eq.py.generated_problems)
