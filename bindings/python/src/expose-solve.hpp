//
// Copyright (c) 2022-2024 INRIA
//
#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
solveDenseQp(nanobind::module_ m)
{
  m.def(
    "solve",
    nanobind::overload_cast<optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<bool>,
                            bool,
                            bool,
                            optional<isize>,
                            proxsuite::proxqp::InitialGuessStatus,
                            bool,
                            optional<T>,
                            optional<T>,
                            bool,
                            optional<T>>(&dense::solve<T>),
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H") = nullopt,
    nanobind::arg("g") = nullopt,
    nanobind::arg("A") = nullopt,
    nanobind::arg("b") = nullopt,
    nanobind::arg("C") = nullopt,
    nanobind::arg("l") = nullopt,
    nanobind::arg("u") = nullopt,
    nanobind::arg("x") = nullopt,
    nanobind::arg("y") = nullopt,
    nanobind::arg("z") = nullopt,
    nanobind::arg("eps_abs") = nullopt,
    nanobind::arg("eps_rel") = nullopt,
    nanobind::arg("rho") = nullopt,
    nanobind::arg("mu_eq") = nullopt,
    nanobind::arg("mu_in") = nullopt,
    nanobind::arg("verbose") = nullopt,
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nullopt,
    nanobind::arg("initial_guess") =
      InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nullopt,
    nanobind::arg("eps_duality_gap_rel") = nullopt,
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.);

  m.def(
    "solve",
    nanobind::overload_cast<optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<bool>,
                            bool,
                            bool,
                            optional<isize>,
                            proxsuite::proxqp::InitialGuessStatus,
                            bool,
                            optional<T>,
                            optional<T>,
                            bool,
                            optional<T>>(&dense::solve<T>),
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H") = nullopt,
    nanobind::arg("g") = nullopt,
    nanobind::arg("A") = nullopt,
    nanobind::arg("b") = nullopt,
    nanobind::arg("C") = nullopt,
    nanobind::arg("l") = nullopt,
    nanobind::arg("u") = nullopt,
    nanobind::arg("l_box") = nullopt,
    nanobind::arg("u_box") = nullopt,
    nanobind::arg("x") = nullopt,
    nanobind::arg("y") = nullopt,
    nanobind::arg("z") = nullopt,
    nanobind::arg("eps_abs") = nullopt,
    nanobind::arg("eps_rel") = nullopt,
    nanobind::arg("rho") = nullopt,
    nanobind::arg("mu_eq") = nullopt,
    nanobind::arg("mu_in") = nullopt,
    nanobind::arg("verbose") = nullopt,
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nullopt,
    nanobind::arg("initial_guess") =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nullopt,
    nanobind::arg("eps_duality_gap_rel") = nullopt,
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.);
}

} // namespace python
} // namespace dense

namespace sparse {
namespace python {

template<typename T, typename I>
void
solveSparseQp(nanobind::module_ m)
{
  m.def(
    "solve",
    &sparse::solve<T, I>,
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H") = nullopt,
    nanobind::arg("g") = nullopt,
    nanobind::arg("A") = nullopt,
    nanobind::arg("b") = nullopt,
    nanobind::arg("C") = nullopt,
    nanobind::arg("l") = nullopt,
    nanobind::arg("u") = nullopt,
    nanobind::arg("x") = nullopt,
    nanobind::arg("y") = nullopt,
    nanobind::arg("z") = nullopt,
    nanobind::arg("eps_abs") = nullopt,
    nanobind::arg("eps_rel") = nullopt,
    nanobind::arg("rho") = nullopt,
    nanobind::arg("mu_eq") = nullopt,
    nanobind::arg("mu_in") = nullopt,
    nanobind::arg("verbose") = nullopt,
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nullopt,
    nanobind::arg("initial_guess") =
      InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("sparse_backend") = SparseBackend::Automatic,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nullopt,
    nanobind::arg("eps_duality_gap_rel") = nullopt,
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.);
}

} // namespace python
} // namespace sparse
} // namespace proxqp
} // namespace proxsuite
