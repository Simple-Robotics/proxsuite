include_directories(include)

set(LDLT_BENCH_LIBS inria_ldlt inria_ldlt_util)

add_executable(.bench.qp_eq src/qp_eq.cpp)
target_link_libraries(.bench.qp_eq ${LDLT_BENCH_LIBS} CONAN_PKG::benchmark)

add_executable(.bench.qp_in src/qp_in.cpp)
target_link_libraries(.bench.qp_in ${LDLT_BENCH_LIBS} CONAN_PKG::benchmark)

add_executable(bench-parser src/parser.cpp)
target_link_libraries(bench-parser PUBLIC CONAN_PKG::eigen inria_ldlt)
get_target_property(PARSER_INCLUDE_DIR bench-parser INCLUDE_DIRECTORIES)
message(STATUS "PARSER_INCLUDE_DIR: e${PARSER_INCLUDE_DIR}e")

add_executable(.bench.osqp src/osqp.cpp)
target_link_libraries(
  .bench.osqp
  ${LDLT_BENCH_LIBS}
  CONAN_PKG::benchmark
  osqpstatic
)

add_executable(.bench.osqp.py src/osqp_py.cpp)
target_link_libraries(.bench.osqp.py ${LDLT_BENCH_LIBS} osqpstatic)
target_compile_definitions(
  .bench.osqp.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${INRIA_LDLT_QP_PYTHON_PATH}"
)

add_executable(.bench.qpalm.py src/qpalm_py.cpp)
target_link_libraries(.bench.qpalm.py ${LDLT_BENCH_LIBS} qpalm)
target_compile_definitions(
  .bench.qpalm.py
  PUBLIC INRIA_LDLT_QP_PYTHON_PATH="${INRIA_LDLT_QP_PYTHON_PATH}"
)
