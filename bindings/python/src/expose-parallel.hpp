//
// Copyright (c) 2023 INRIA
//
#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <proxsuite/proxqp/parallel/qp_solve.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h> // For binding STL containers

PYBIND11_MAKE_OPAQUE(std::vector<proxsuite::proxqp::dense::QP<double>>);

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
solveDenseQpParallel(pybind11::module_ m)
{
  pybind11::bind_vector<std::vector<proxsuite::proxqp::dense::QP<T>>>(
    m, "VectorDenseQP");

  m.def("solve_in_parallel",
        &parallel::qp_solve_in_parallel<T>,
        "Function for solving a list of dense QPs in parallel.",
        pybind11::arg_v("num_threads",
                        "number of threads used for the computation."),
        pybind11::arg_v("qps", "List of initialized dense Qps.")
  );
}

} // namespace python
} // namespace dense

} // namespace proxqp
} // namespace proxsuite
