//
// Copyright (c) 2023 INRIA
//

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <proxsuite/proxqp/parallel/qp_solve.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/bind_vector.h>

NB_MAKE_OPAQUE(std::vector<proxsuite::proxqp::dense::QP<double>>)
NB_MAKE_OPAQUE(std::vector<proxsuite::proxqp::dense::Vec<double>>)
namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
solveDenseQpParallel(nanobind::module_ m)
{

  nanobind::bind_vector<std::vector<proxsuite::proxqp::dense::Vec<T>>>(
    m, "VectorLossDerivatives");

  nanobind::bind_vector<std::vector<proxsuite::proxqp::dense::QP<T>>>(
    m, "VectorQP");

  m.def("solve_in_parallel",
        nanobind::overload_cast<std::vector<proxqp::dense::QP<T>>&,
                                const optional<size_t>>(&solve_in_parallel<T>),
        "Function for solving a list of dense QPs in parallel.",
        nanobind::arg("qps"),
        nanobind::arg("num_threads") = nullopt);

  m.def(
    "solve_in_parallel",
    nanobind::overload_cast<proxqp::dense::BatchQP<T>&, const optional<size_t>>(
      &solve_in_parallel<T>),
    "Function for solving a list of dense QPs in parallel.",
    nanobind::arg("qps"),
    nanobind::arg("num_threads") = nullopt);

  // m.def("solve_in_parallel",
  //       &qp_solve_in_parallel<T>,
  //       "Function for solving a list of dense QPs in parallel.",
  //       nanobind::arg("num_threads") = nullopt,
  //       nanobind::arg("qps"));

  m.def("solve_backward_in_parallel",
        nanobind::overload_cast<optional<const size_t>,
                                proxqp::dense::BatchQP<T>&,
                                std::vector<proxqp::dense::Vec<T>>&,
                                T,
                                T,
                                T>(&qp_solve_backward_in_parallel<T>),
        "Function for solving a list of dense QPs in parallel.",
        nanobind::arg("num_threads") = nullopt,
        nanobind::arg("qps"),
        nanobind::arg("loss_derivatives"),
        nanobind::arg("eps") = 1e-4,
        nanobind::arg("rho_backward") = 1e-6,
        nanobind::arg("mu_backward") = 1e-6);

  m.def("solve_backward_in_parallel",
        nanobind::overload_cast<optional<const size_t>,
                                std::vector<proxqp::dense::QP<T>>&,
                                std::vector<proxqp::dense::Vec<T>>&,
                                T,
                                T,
                                T>(&qp_solve_backward_in_parallel<T>),
        "Function for solving a list of dense QPs in parallel.",
        nanobind::arg("num_threads") = nullopt,
        nanobind::arg("qps"),
        nanobind::arg("loss_derivatives"),
        nanobind::arg("eps") = 1e-4,
        nanobind::arg("rho_backward") = 1e-6,
        nanobind::arg("mu_backward") = 1e-6);
}

} // namespace python
} // namespace dense

namespace sparse {
namespace python {
template<typename T, typename I>
void
solveSparseQpParallel(nanobind::module_ m)
{
  m.def(
    "solve_in_parallel",
    nanobind::overload_cast<proxqp::sparse::BatchQP<T, I>&,
                            const optional<size_t>>(&solve_in_parallel<T, I>),
    "Function for solving a list of sparse QPs in parallel.",
    nanobind::arg("qps"),
    nanobind::arg("num_threads") = nullopt);
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
