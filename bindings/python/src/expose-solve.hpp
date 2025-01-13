//
// Copyright (c) 2022-2024 INRIA
//
#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include "optional-eigen-fix.hpp"

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
    "Function for solving a QP problem using PROXQP dense backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H"),
    nanobind::arg("g"),
    nanobind::arg("A").none(),
    nanobind::arg("b").none(),
    nanobind::arg("C").none(),
    nanobind::arg("l").none(),
    nanobind::arg("u").none(),
    nanobind::arg("x") = nanobind::none(),
    nanobind::arg("y") = nanobind::none(),
    nanobind::arg("z") = nanobind::none(),
    nanobind::arg("eps_abs") = nanobind::none(),
    nanobind::arg("eps_rel") = nanobind::none(),
    nanobind::arg("rho") = nanobind::none(),
    nanobind::arg("mu_eq") = nanobind::none(),
    nanobind::arg("mu_in") = nanobind::none(),
    nanobind::arg("verbose") = nanobind::none(),
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nanobind::none(),
    nanobind::arg("initial_guess") =
      InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nanobind::none(),
    nanobind::arg("eps_duality_gap_rel") = nanobind::none(),
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
    "Function for solving a QP problem using PROXQP dense backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H"),
    nanobind::arg("g"),
    nanobind::arg("A") = nanobind::none(),
    nanobind::arg("b") = nanobind::none(),
    nanobind::arg("C") = nanobind::none(),
    nanobind::arg("l") = nanobind::none(),
    nanobind::arg("u") = nanobind::none(),
    nanobind::arg("l_box") = nanobind::none(),
    nanobind::arg("u_box") = nanobind::none(),
    nanobind::arg("x") = nanobind::none(),
    nanobind::arg("y") = nanobind::none(),
    nanobind::arg("z") = nanobind::none(),
    nanobind::arg("eps_abs") = nanobind::none(),
    nanobind::arg("eps_rel") = nanobind::none(),
    nanobind::arg("rho") = nanobind::none(),
    nanobind::arg("mu_eq") = nanobind::none(),
    nanobind::arg("mu_in") = nanobind::none(),
    nanobind::arg("verbose") = nanobind::none(),
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nanobind::none(),
    nanobind::arg("initial_guess") =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nanobind::none(),
    nanobind::arg("eps_duality_gap_rel") = nanobind::none(),
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.);

  m.def("solve_no_gil",
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
        "Function for solving a QP problem using PROXQP dense backend directly "
        "without defining a QP object and while releasing the Global "
        "Interpreter Lock (GIL). "
        "It is possible to set up some of the solver "
        "parameters (warm start, initial guess option, proximal step sizes, "
        "absolute and relative accuracies, maximum number of iterations, "
        "preconditioner execution).",
        nanobind::arg("H"),
        nanobind::arg("g"),
        nanobind::arg("A").none(),
        nanobind::arg("b").none(),
        nanobind::arg("C").none(),
        nanobind::arg("l").none(),
        nanobind::arg("u").none(),
        nanobind::arg("x") = nanobind::none(),
        nanobind::arg("y") = nanobind::none(),
        nanobind::arg("z") = nanobind::none(),
        nanobind::arg("eps_abs") = nanobind::none(),
        nanobind::arg("eps_rel") = nanobind::none(),
        nanobind::arg("rho") = nanobind::none(),
        nanobind::arg("mu_eq") = nanobind::none(),
        nanobind::arg("mu_in") = nanobind::none(),
        nanobind::arg("verbose") = nanobind::none(),
        nanobind::arg("compute_preconditioner") = true,
        nanobind::arg("compute_timings") = false,
        nanobind::arg("max_iter") = nanobind::none(),
        nanobind::arg("initial_guess") =
          InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
        nanobind::arg("check_duality_gap") = false,
        nanobind::arg("eps_duality_gap_abs") = nanobind::none(),
        nanobind::arg("eps_duality_gap_rel") = nanobind::none(),
        nanobind::arg("primal_infeasibility_solving") = false,
        nanobind::arg("default_H_eigenvalue_estimate") = 0.,
        nanobind::call_guard<nanobind::gil_scoped_release>());

  m.def(
    "solve_no_gil",
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
    "Function for solving a QP problem using PROXQP dense backend directly "
    "without defining a QP object and while releasing the Global Interpreter "
    "Lock (GIL). "
    "It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H"),
    nanobind::arg("g"),
    nanobind::arg("A") = nanobind::none(),
    nanobind::arg("b") = nanobind::none(),
    nanobind::arg("C") = nanobind::none(),
    nanobind::arg("l") = nanobind::none(),
    nanobind::arg("u") = nanobind::none(),
    nanobind::arg("l_box") = nanobind::none(),
    nanobind::arg("u_box") = nanobind::none(),
    nanobind::arg("x") = nanobind::none(),
    nanobind::arg("y") = nanobind::none(),
    nanobind::arg("z") = nanobind::none(),
    nanobind::arg("eps_abs") = nanobind::none(),
    nanobind::arg("eps_rel") = nanobind::none(),
    nanobind::arg("rho") = nanobind::none(),
    nanobind::arg("mu_eq") = nanobind::none(),
    nanobind::arg("mu_in") = nanobind::none(),
    nanobind::arg("verbose") = nanobind::none(),
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nanobind::none(),
    nanobind::arg("initial_guess") =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nanobind::none(),
    nanobind::arg("eps_duality_gap_rel") = nanobind::none(),
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.,
    nanobind::call_guard<nanobind::gil_scoped_release>());
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
    nanobind::arg("H") = nanobind::none(),
    nanobind::arg("g") = nanobind::none(),
    nanobind::arg("A") = nanobind::none(),
    nanobind::arg("b") = nanobind::none(),
    nanobind::arg("C") = nanobind::none(),
    nanobind::arg("l") = nanobind::none(),
    nanobind::arg("u") = nanobind::none(),
    nanobind::arg("x") = nanobind::none(),
    nanobind::arg("y") = nanobind::none(),
    nanobind::arg("z") = nanobind::none(),
    nanobind::arg("eps_abs") = nanobind::none(),
    nanobind::arg("eps_rel") = nanobind::none(),
    nanobind::arg("rho") = nanobind::none(),
    nanobind::arg("mu_eq") = nanobind::none(),
    nanobind::arg("mu_in") = nanobind::none(),
    nanobind::arg("verbose") = nanobind::none(),
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nanobind::none(),
    nanobind::arg("initial_guess") =
      InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("sparse_backend") = SparseBackend::Automatic,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nanobind::none(),
    nanobind::arg("eps_duality_gap_rel") = nanobind::none(),
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.);

  m.def(
    "solve_no_gil",
    &sparse::solve<T, I>,
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object and while releasing the Global Interpreter "
    "Lock (GIL). "
    "It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    nanobind::arg("H") = nanobind::none(),
    nanobind::arg("g") = nanobind::none(),
    nanobind::arg("A") = nanobind::none(),
    nanobind::arg("b") = nanobind::none(),
    nanobind::arg("C") = nanobind::none(),
    nanobind::arg("l") = nanobind::none(),
    nanobind::arg("u") = nanobind::none(),
    nanobind::arg("x") = nanobind::none(),
    nanobind::arg("y") = nanobind::none(),
    nanobind::arg("z") = nanobind::none(),
    nanobind::arg("eps_abs") = nanobind::none(),
    nanobind::arg("eps_rel") = nanobind::none(),
    nanobind::arg("rho") = nanobind::none(),
    nanobind::arg("mu_eq") = nanobind::none(),
    nanobind::arg("mu_in") = nanobind::none(),
    nanobind::arg("verbose") = nanobind::none(),
    nanobind::arg("compute_preconditioner") = true,
    nanobind::arg("compute_timings") = false,
    nanobind::arg("max_iter") = nanobind::none(),
    nanobind::arg("initial_guess") =
      InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    nanobind::arg("sparse_backend") = SparseBackend::Automatic,
    nanobind::arg("check_duality_gap") = false,
    nanobind::arg("eps_duality_gap_abs") = nanobind::none(),
    nanobind::arg("eps_duality_gap_rel") = nanobind::none(),
    nanobind::arg("primal_infeasibility_solving") = false,
    nanobind::arg("default_H_eigenvalue_estimate") = 0.,
    nanobind::call_guard<nanobind::gil_scoped_release>());
}

} // namespace python
} // namespace sparse
} // namespace proxqp
} // namespace proxsuite
